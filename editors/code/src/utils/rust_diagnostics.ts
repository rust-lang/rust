import * as path from 'path';
import * as vscode from 'vscode';

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
    suggested_replacement?: string;
    suggestion_applicability?:
        | 'MachineApplicable'
        | 'HasPlaceholders'
        | 'MaybeIncorrect'
        | 'Unspecified';
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
    codeActions: vscode.CodeAction[];
}

interface MappedRustChildDiagnostic {
    related?: vscode.DiagnosticRelatedInformation;
    codeAction?: vscode.CodeAction;
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
 * Converts a Rust span to a VsCode location
 */
function mapSpanToLocation(span: RustDiagnosticSpan): vscode.Location {
    const fileName = path.join(vscode.workspace.rootPath!, span.file_name);
    const fileUri = vscode.Uri.file(fileName);

    const range = new vscode.Range(
        new vscode.Position(span.line_start - 1, span.column_start - 1),
        new vscode.Position(span.line_end - 1, span.column_end - 1)
    );

    return new vscode.Location(fileUri, range);
}

/**
 * Converts a secondary Rust span to a VsCode related information
 *
 * If the span is unlabelled this will return `undefined`.
 */
function mapSecondarySpanToRelated(
    span: RustDiagnosticSpan
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

    const { code } = rd.code;
    return code.startsWith('unused_') || code === 'dead_code';
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
        const edit = new vscode.WorkspaceEdit();
        edit.replace(location.uri, location.range, span.suggested_replacement);

        // Include our replacement in the label unless it's empty
        const title = span.suggested_replacement
            ? `${rd.message}: \`${span.suggested_replacement}\``
            : rd.message;

        const codeAction = new vscode.CodeAction(
            title,
            vscode.CodeActionKind.QuickFix
        );

        codeAction.edit = edit;
        codeAction.isPreferred =
            span.suggestion_applicability === 'MachineApplicable';

        return { codeAction };
    } else {
        const related = new vscode.DiagnosticRelatedInformation(
            location,
            rd.message
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
 * 3. Categorising child diagnostics as either Quick Fix actions,
 *    `relatedInformation` or additional message lines.
 *
 * If the diagnostic has no primary span this will return `undefined`
 */
export function mapRustDiagnosticToVsCode(
    rd: RustDiagnostic
): MappedRustDiagnostic | undefined {
    const codeActions = [];

    const primarySpan = rd.spans.find(s => s.is_primary);
    if (!primarySpan) {
        return;
    }

    const location = mapSpanToLocation(primarySpan);
    const secondarySpans = rd.spans.filter(s => !s.is_primary);

    const severity = mapLevelToSeverity(rd.level);

    const vd = new vscode.Diagnostic(location.range, rd.message, severity);

    vd.source = 'rustc';
    vd.code = rd.code ? rd.code.code : undefined;
    vd.relatedInformation = [];

    for (const secondarySpan of secondarySpans) {
        const related = mapSecondarySpanToRelated(secondarySpan);
        if (related) {
            vd.relatedInformation.push(related);
        }
    }

    for (const child of rd.children) {
        const { related, codeAction, messageLine } = mapRustChildDiagnostic(
            child
        );

        if (related) {
            vd.relatedInformation.push(related);
        }
        if (codeAction) {
            codeActions.push(codeAction);
        }
        if (messageLine) {
            vd.message += `\n${messageLine}`;
        }
    }

    if (isUnusedOrUnnecessary(rd)) {
        vd.tags = [vscode.DiagnosticTag.Unnecessary];
    }

    return {
        location,
        diagnostic: vd,
        codeActions
    };
}
