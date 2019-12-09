import * as assert from 'assert';
import * as vscode from 'vscode';

import { SuggestionApplicability } from '../../../utils/diagnostics/rust';
import SuggestedFix from '../../../utils/diagnostics/SuggestedFix';

const location1 = new vscode.Location(
    vscode.Uri.file('/file/1'),
    new vscode.Range(new vscode.Position(1, 2), new vscode.Position(3, 4)),
);

const location2 = new vscode.Location(
    vscode.Uri.file('/file/2'),
    new vscode.Range(new vscode.Position(5, 6), new vscode.Position(7, 8)),
);

describe('SuggestedFix', () => {
    describe('isEqual', () => {
        it('should treat identical instances as equal', () => {
            const suggestion1 = new SuggestedFix(
                'Replace me!',
                location1,
                'With this!',
            );

            const suggestion2 = new SuggestedFix(
                'Replace me!',
                location1,
                'With this!',
            );

            assert(suggestion1.isEqual(suggestion2));
        });

        it('should treat instances with different titles as inequal', () => {
            const suggestion1 = new SuggestedFix(
                'Replace me!',
                location1,
                'With this!',
            );

            const suggestion2 = new SuggestedFix(
                'Not the same title!',
                location1,
                'With this!',
            );

            assert(!suggestion1.isEqual(suggestion2));
        });

        it('should treat instances with different replacements as inequal', () => {
            const suggestion1 = new SuggestedFix(
                'Replace me!',
                location1,
                'With this!',
            );

            const suggestion2 = new SuggestedFix(
                'Replace me!',
                location1,
                'With something else!',
            );

            assert(!suggestion1.isEqual(suggestion2));
        });

        it('should treat instances with different locations as inequal', () => {
            const suggestion1 = new SuggestedFix(
                'Replace me!',
                location1,
                'With this!',
            );

            const suggestion2 = new SuggestedFix(
                'Replace me!',
                location2,
                'With this!',
            );

            assert(!suggestion1.isEqual(suggestion2));
        });

        it('should treat instances with different applicability as inequal', () => {
            const suggestion1 = new SuggestedFix(
                'Replace me!',
                location1,
                'With this!',
                SuggestionApplicability.MachineApplicable,
            );

            const suggestion2 = new SuggestedFix(
                'Replace me!',
                location2,
                'With this!',
                SuggestionApplicability.HasPlaceholders,
            );

            assert(!suggestion1.isEqual(suggestion2));
        });
    });

    describe('toCodeAction', () => {
        it('should map a simple suggestion', () => {
            const suggestion = new SuggestedFix(
                'Replace me!',
                location1,
                'With this!',
            );

            const codeAction = suggestion.toCodeAction();
            assert.strictEqual(codeAction.kind, vscode.CodeActionKind.QuickFix);
            assert.strictEqual(codeAction.title, 'Replace me!');
            assert.strictEqual(codeAction.isPreferred, false);

            const edit = codeAction.edit;
            if (!edit) {
                return assert.fail('Code Action edit unexpectedly missing');
            }

            const editEntries = edit.entries();
            assert.strictEqual(editEntries.length, 1);

            const [[editUri, textEdits]] = editEntries;
            assert.strictEqual(editUri.toString(), location1.uri.toString());

            assert.strictEqual(textEdits.length, 1);
            const [textEdit] = textEdits;

            assert(textEdit.range.isEqual(location1.range));
            assert.strictEqual(textEdit.newText, 'With this!');
        });
    });
});
