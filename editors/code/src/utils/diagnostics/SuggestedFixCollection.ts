import * as vscode from 'vscode';
import SuggestedFix from './SuggestedFix';

/**
 * Collection of suggested fixes across multiple documents
 *
 * This stores `SuggestedFix` model objects and returns them via the
 * `vscode.CodeActionProvider` interface.
 */
export default class SuggestedFixCollection
    implements vscode.CodeActionProvider {
    public static PROVIDED_CODE_ACTION_KINDS = [vscode.CodeActionKind.QuickFix];

    /**
     * Map of document URI strings to suggested fixes
     */
    private suggestedFixes: Map<string, SuggestedFix[]>;

    constructor() {
        this.suggestedFixes = new Map();
    }

    /**
     * Clears all suggested fixes across all documents
     */
    public clear(): void {
        this.suggestedFixes = new Map();
    }

    /**
     * Adds a suggested fix for the given diagnostic
     *
     * Some suggested fixes will appear in multiple diagnostics. For example,
     * forgetting a `mut` on a variable will suggest changing the delaration on
     * every mutable usage site. If the suggested fix has already been added
     * this method will instead associate the existing fix with the new
     * diagnostic.
     */
    public addSuggestedFixForDiagnostic(
        suggestedFix: SuggestedFix,
        diagnostic: vscode.Diagnostic,
    ): void {
        const fileUriString = suggestedFix.location.uri.toString();
        const fileSuggestions = this.suggestedFixes.get(fileUriString) || [];

        const existingSuggestion = fileSuggestions.find(s =>
            s.isEqual(suggestedFix),
        );

        if (existingSuggestion) {
            // The existing suggestion also applies to this new diagnostic
            existingSuggestion.diagnostics.push(diagnostic);
        } else {
            // We haven't seen this suggestion before
            suggestedFix.diagnostics.push(diagnostic);
            fileSuggestions.push(suggestedFix);
        }

        this.suggestedFixes.set(fileUriString, fileSuggestions);
    }

    /**
     * Filters suggested fixes by their document and range and converts them to
     * code actions
     */
    public provideCodeActions(
        document: vscode.TextDocument,
        range: vscode.Range,
    ): vscode.CodeAction[] {
        const documentUriString = document.uri.toString();

        const suggestedFixes = this.suggestedFixes.get(documentUriString);
        return (suggestedFixes || [])
            .filter(({ location }) => location.range.intersection(range))
            .map(suggestedEdit => suggestedEdit.toCodeAction());
    }
}
