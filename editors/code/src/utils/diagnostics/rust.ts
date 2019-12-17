import * as path from 'path';
import * as vscode from 'vscode';

import SuggestedFix from './SuggestedFix';

export enum SuggestionApplicability {
    MachineApplicable = 'MachineApplicable',
    HasPlaceholders = 'HasPlaceholders',
    MaybeIncorrect = 'MaybeIncorrect',
    Unspecified = 'Unspecified',
}

export interface RustDiagnosticSpanMacroExpansion {
    span: RustDiagnosticSpan;
    macro_decl_name: string;
    def_site_span?: RustDiagnosticSpan;
}

// Reference:
// https://github.com/rust-lang/rust/blob/master/src/libsyntax/json.rs
export interface RustDiagnosticSpan {
    line_start: number;
    line_end: number;
    column_start: number;
    column_end: number;
    is_primary: boolean;
    file_name: string;
    label?: string;
    expansion?: RustDiagnosticSpanMacroExpansion;
    suggested_replacement?: string;
    suggestion_applicability?: SuggestionApplicability;
}

export interface RustDiagnostic {
    spans: RustDiagnosticSpan[];
    rendered: string;
    message: string;
    level: string;
    code?: {
        code: string;
    };
    children: RustDiagnostic[];
}

export interface MappedRustDiagnostic {
    location: vscode.Location;
    diagnostic: vscode.Diagnostic;
    suggestedFixes: SuggestedFix[];
}

interface MappedRustChildDiagnostic {
    related?: vscode.DiagnosticRelatedInformation;
    suggestedFix?: SuggestedFix;
    messageLine?: string;
}

/**
 * Converts a Rust level string to a VsCode severity
 */
function mapLevelToSeverity(s: string): vscode.DiagnosticSeverity {
    if (s === 'error') {
        return vscode.DiagnosticSeverity.Error;
    }
    if (s.startsWith('warn')) {
        return vscode.DiagnosticSeverity.Warning;
    }
    return vscode.DiagnosticSeverity.Information;
}

/**
 * Check whether a file name is from macro invocation
 */
function isFromMacro(fileName: string): boolean {
    return fileName.startsWith('<') && fileName.endsWith('>');
}

/**
 * Converts a Rust macro span to a VsCode location recursively
 */
function mapMacroSpanToLocation(
    spanMacro: RustDiagnosticSpanMacroExpansion,
): vscode.Location | undefined {
    if (!isFromMacro(spanMacro.span.file_name)) {
        return mapSpanToLocation(spanMacro.span);
    }

    if (spanMacro.span.expansion) {
        return mapMacroSpanToLocation(spanMacro.span.expansion);
    }

    return;
}

/**
 * Converts a Rust span to a VsCode location
 */
function mapSpanToLocation(span: RustDiagnosticSpan): vscode.Location {
    if (isFromMacro(span.file_name) && span.expansion) {
        const macroLoc = mapMacroSpanToLocation(span.expansion);
        if (macroLoc) {
            return macroLoc;
        }
    }

    const fileName = path.join(vscode.workspace.rootPath || '', span.file_name);
    const fileUri = vscode.Uri.file(fileName);

    const range = new vscode.Range(
        new vscode.Position(span.line_start - 1, span.column_start - 1),
        new vscode.Position(span.line_end - 1, span.column_end - 1),
    );

    return new vscode.Location(fileUri, range);
}

/**
 * Converts a secondary Rust span to a VsCode related information
 *
 * If the span is unlabelled this will return `undefined`.
 */
function mapSecondarySpanToRelated(
    span: RustDiagnosticSpan,
): vscode.DiagnosticRelatedInformation | undefined {
    if (!span.label) {
        // Nothing to label this with
        return;
    }

    const location = mapSpanToLocation(span);
    return new vscode.DiagnosticRelatedInformation(location, span.label);
}

/**
 * Determines if diagnostic is related to unused code
 */
function isUnusedOrUnnecessary(rd: RustDiagnostic): boolean {
    if (!rd.code) {
        return false;
    }

    return [
        'dead_code',
        'unknown_lints',
        'unreachable_code',
        'unused_attributes',
        'unused_imports',
        'unused_macros',
        'unused_variables',
    ].includes(rd.code.code);
}

/**
 * Determines if diagnostic is related to deprecated code
 */
function isDeprecated(rd: RustDiagnostic): boolean {
    if (!rd.code) {
        return false;
    }

    return ['deprecated'].includes(rd.code.code);
}

/**
 * Converts a Rust child diagnostic to a VsCode related information
 *
 * This can have three outcomes:
 *
 * 1. If this is no primary span this will return a `noteLine`
 * 2. If there is a primary span with a suggested replacement it will return a
 *    `codeAction`.
 * 3. If there is a primary span without a suggested replacement it will return
 *    a `related`.
 */
function mapRustChildDiagnostic(rd: RustDiagnostic): MappedRustChildDiagnostic {
    const span = rd.spans.find(s => s.is_primary);

    if (!span) {
        // `rustc` uses these spanless children as a way to print multi-line
        // messages
        return { messageLine: rd.message };
    }

    // If we have a primary span use its location, otherwise use the parent
    const location = mapSpanToLocation(span);

    // We need to distinguish `null` from an empty string
    if (span && typeof span.suggested_replacement === 'string') {
        // Include our replacement in the title unless it's empty
        const title = span.suggested_replacement
            ? `${rd.message}: \`${span.suggested_replacement}\``
            : rd.message;

        return {
            suggestedFix: new SuggestedFix(
                title,
                location,
                span.suggested_replacement,
                span.suggestion_applicability,
            ),
        };
    } else {
        const related = new vscode.DiagnosticRelatedInformation(
            location,
            rd.message,
        );

        return { related };
    }
}

/**
 * Converts a Rust root diagnostic to VsCode form
 *
 * This flattens the Rust diagnostic by:
 *
 * 1. Creating a `vscode.Diagnostic` with the root message and primary span.
 * 2. Adding any labelled secondary spans to `relatedInformation`
 * 3. Categorising child diagnostics as either `SuggestedFix`es,
 *    `relatedInformation` or additional message lines.
 *
 * If the diagnostic has no primary span this will return `undefined`
 */
export function mapRustDiagnosticToVsCode(
    rd: RustDiagnostic,
): MappedRustDiagnostic | undefined {
    const primarySpan = rd.spans.find(s => s.is_primary);
    if (!primarySpan) {
        return;
    }

    const location = mapSpanToLocation(primarySpan);
    const secondarySpans = rd.spans.filter(s => !s.is_primary);

    const severity = mapLevelToSeverity(rd.level);
    let primarySpanLabel = primarySpan.label;

    const vd = new vscode.Diagnostic(location.range, rd.message, severity);

    let source = 'rustc';
    let code = rd.code && rd.code.code;
    if (code) {
        // See if this is an RFC #2103 scoped lint (e.g. from Clippy)
        const scopedCode = code.split('::');
        if (scopedCode.length === 2) {
            [source, code] = scopedCode;
        }
    }

    vd.source = source;
    vd.code = code;
    vd.relatedInformation = [];
    vd.tags = [];

    for (const secondarySpan of secondarySpans) {
        const related = mapSecondarySpanToRelated(secondarySpan);
        if (related) {
            vd.relatedInformation.push(related);
        }
    }

    const suggestedFixes = [];
    for (const child of rd.children) {
        const { related, suggestedFix, messageLine } = mapRustChildDiagnostic(
            child,
        );

        if (related) {
            vd.relatedInformation.push(related);
        }
        if (suggestedFix) {
            suggestedFixes.push(suggestedFix);
        }
        if (messageLine) {
            vd.message += `\n${messageLine}`;

            // These secondary messages usually duplicate the content of the
            // primary span label.
            primarySpanLabel = undefined;
        }
    }

    if (primarySpanLabel) {
        vd.message += `\n${primarySpanLabel}`;
    }

    if (isUnusedOrUnnecessary(rd)) {
        vd.tags.push(vscode.DiagnosticTag.Unnecessary);
    }

    if (isDeprecated(rd)) {
        vd.tags.push(vscode.DiagnosticTag.Deprecated);
    }

    return {
        location,
        diagnostic: vd,
        suggestedFixes,
    };
}
