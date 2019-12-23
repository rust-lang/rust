import * as vscode from 'vscode';

import { SuggestionApplicability } from './rust';

/**
 * Model object for text replacements suggested by the Rust compiler
 *
 * This is an intermediate form between the raw `rustc` JSON and a
 * `vscode.CodeAction`. It's optimised for the use-cases of
 * `SuggestedFixCollection`.
 */
export default class SuggestedFix {
    public readonly title: string;
    public readonly location: vscode.Location;
    public readonly replacement: string;
    public readonly applicability: SuggestionApplicability;

    /**
     * Diagnostics this suggested fix could resolve
     */
    public diagnostics: vscode.Diagnostic[];

    constructor(
        title: string,
        location: vscode.Location,
        replacement: string,
        applicability: SuggestionApplicability = SuggestionApplicability.Unspecified,
    ) {
        this.title = title;
        this.location = location;
        this.replacement = replacement;
        this.applicability = applicability;
        this.diagnostics = [];
    }

    /**
     * Determines if this suggested fix is equivalent to another instance
     */
    public isEqual(other: SuggestedFix): boolean {
        return (
            this.title === other.title &&
            this.location.range.isEqual(other.location.range) &&
            this.replacement === other.replacement &&
            this.applicability === other.applicability
        );
    }

    /**
     * Converts this suggested fix to a VS Code Quick Fix code action
     */
    public toCodeAction(): vscode.CodeAction {
        const codeAction = new vscode.CodeAction(
            this.title,
            vscode.CodeActionKind.QuickFix,
        );

        const edit = new vscode.WorkspaceEdit();
        edit.replace(this.location.uri, this.location.range, this.replacement);
        codeAction.edit = edit;

        codeAction.isPreferred =
            this.applicability === SuggestionApplicability.MachineApplicable;

        codeAction.diagnostics = [...this.diagnostics];
        return codeAction;
    }
}
