import * as assert from 'assert';
import * as vscode from 'vscode';

import SuggestedFix from '../../../utils/diagnostics/SuggestedFix';
import SuggestedFixCollection from '../../../utils/diagnostics/SuggestedFixCollection';

const uri1 = vscode.Uri.file('/file/1');
const uri2 = vscode.Uri.file('/file/2');

const mockDocument1 = ({
    uri: uri1,
} as unknown) as vscode.TextDocument;

const mockDocument2 = ({
    uri: uri2,
} as unknown) as vscode.TextDocument;

const range1 = new vscode.Range(
    new vscode.Position(1, 2),
    new vscode.Position(3, 4),
);
const range2 = new vscode.Range(
    new vscode.Position(5, 6),
    new vscode.Position(7, 8),
);

const diagnostic1 = new vscode.Diagnostic(range1, 'First diagnostic');
const diagnostic2 = new vscode.Diagnostic(range2, 'Second diagnostic');

// This is a mutable object so return a fresh instance every time
function suggestion1(): SuggestedFix {
    return new SuggestedFix(
        'Replace me!',
        new vscode.Location(uri1, range1),
        'With this!',
    );
}

describe('SuggestedFixCollection', () => {
    it('should add a suggestion then return it as a code action', () => {
        const suggestedFixes = new SuggestedFixCollection();
        suggestedFixes.addSuggestedFixForDiagnostic(suggestion1(), diagnostic1);

        // Specify the document and range that exactly matches
        const codeActions = suggestedFixes.provideCodeActions(
            mockDocument1,
            range1,
        );

        assert.strictEqual(codeActions.length, 1);
        const [codeAction] = codeActions;
        assert.strictEqual(codeAction.title, suggestion1().title);

        const { diagnostics } = codeAction;
        if (!diagnostics) {
            return assert.fail('Diagnostics unexpectedly missing');
        }

        assert.strictEqual(diagnostics.length, 1);
        assert.strictEqual(diagnostics[0], diagnostic1);
    });

    it('should not return code actions for different ranges', () => {
        const suggestedFixes = new SuggestedFixCollection();
        suggestedFixes.addSuggestedFixForDiagnostic(suggestion1(), diagnostic1);

        const codeActions = suggestedFixes.provideCodeActions(
            mockDocument1,
            range2,
        );

        assert(!codeActions || codeActions.length === 0);
    });

    it('should not return code actions for different documents', () => {
        const suggestedFixes = new SuggestedFixCollection();
        suggestedFixes.addSuggestedFixForDiagnostic(suggestion1(), diagnostic1);

        const codeActions = suggestedFixes.provideCodeActions(
            mockDocument2,
            range1,
        );

        assert(!codeActions || codeActions.length === 0);
    });

    it('should not return code actions that have been cleared', () => {
        const suggestedFixes = new SuggestedFixCollection();
        suggestedFixes.addSuggestedFixForDiagnostic(suggestion1(), diagnostic1);
        suggestedFixes.clear();

        const codeActions = suggestedFixes.provideCodeActions(
            mockDocument1,
            range1,
        );

        assert(!codeActions || codeActions.length === 0);
    });

    it('should merge identical suggestions together', () => {
        const suggestedFixes = new SuggestedFixCollection();

        // Add the same suggestion for two diagnostics
        suggestedFixes.addSuggestedFixForDiagnostic(suggestion1(), diagnostic1);
        suggestedFixes.addSuggestedFixForDiagnostic(suggestion1(), diagnostic2);

        const codeActions = suggestedFixes.provideCodeActions(
            mockDocument1,
            range1,
        );

        assert.strictEqual(codeActions.length, 1);
        const [codeAction] = codeActions;
        const { diagnostics } = codeAction;

        if (!diagnostics) {
            return assert.fail('Diagnostics unexpectedly missing');
        }

        // We should be associated with both diagnostics
        assert.strictEqual(diagnostics.length, 2);
        assert.strictEqual(diagnostics[0], diagnostic1);
        assert.strictEqual(diagnostics[1], diagnostic2);
    });
});
