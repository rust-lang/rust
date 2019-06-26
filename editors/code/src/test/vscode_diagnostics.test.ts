import * as assert from 'assert';
import * as vscode from 'vscode';

import {
    areCodeActionsEqual,
    areDiagnosticsEqual
} from '../utils/vscode_diagnostics';

const uri = vscode.Uri.file('/file/1');

const range1 = new vscode.Range(
    new vscode.Position(1, 2),
    new vscode.Position(3, 4)
);

const range2 = new vscode.Range(
    new vscode.Position(5, 6),
    new vscode.Position(7, 8)
);

describe('areDiagnosticsEqual', () => {
    it('should treat identical diagnostics as equal', () => {
        const diagnostic1 = new vscode.Diagnostic(
            range1,
            'Hello, world!',
            vscode.DiagnosticSeverity.Error
        );

        const diagnostic2 = new vscode.Diagnostic(
            range1,
            'Hello, world!',
            vscode.DiagnosticSeverity.Error
        );

        assert(areDiagnosticsEqual(diagnostic1, diagnostic2));
    });

    it('should treat diagnostics with different ranges as inequal', () => {
        const diagnostic1 = new vscode.Diagnostic(
            range1,
            'Hello, world!',
            vscode.DiagnosticSeverity.Error
        );

        const diagnostic2 = new vscode.Diagnostic(
            range2,
            'Hello, world!',
            vscode.DiagnosticSeverity.Error
        );

        assert(!areDiagnosticsEqual(diagnostic1, diagnostic2));
    });

    it('should treat diagnostics with different messages as inequal', () => {
        const diagnostic1 = new vscode.Diagnostic(
            range1,
            'Hello, world!',
            vscode.DiagnosticSeverity.Error
        );

        const diagnostic2 = new vscode.Diagnostic(
            range1,
            'Goodbye!, world!',
            vscode.DiagnosticSeverity.Error
        );

        assert(!areDiagnosticsEqual(diagnostic1, diagnostic2));
    });

    it('should treat diagnostics with different severities as inequal', () => {
        const diagnostic1 = new vscode.Diagnostic(
            range1,
            'Hello, world!',
            vscode.DiagnosticSeverity.Warning
        );

        const diagnostic2 = new vscode.Diagnostic(
            range1,
            'Hello, world!',
            vscode.DiagnosticSeverity.Error
        );

        assert(!areDiagnosticsEqual(diagnostic1, diagnostic2));
    });
});

describe('areCodeActionsEqual', () => {
    it('should treat identical actions as equal', () => {
        const codeAction1 = new vscode.CodeAction(
            'Fix me!',
            vscode.CodeActionKind.QuickFix
        );

        const codeAction2 = new vscode.CodeAction(
            'Fix me!',
            vscode.CodeActionKind.QuickFix
        );

        const edit = new vscode.WorkspaceEdit();
        edit.replace(uri, range1, 'Replace with this');
        codeAction1.edit = edit;
        codeAction2.edit = edit;

        assert(areCodeActionsEqual(codeAction1, codeAction2));
    });

    it('should treat actions with different types as inequal', () => {
        const codeAction1 = new vscode.CodeAction(
            'Fix me!',
            vscode.CodeActionKind.Refactor
        );

        const codeAction2 = new vscode.CodeAction(
            'Fix me!',
            vscode.CodeActionKind.QuickFix
        );

        const edit = new vscode.WorkspaceEdit();
        edit.replace(uri, range1, 'Replace with this');
        codeAction1.edit = edit;
        codeAction2.edit = edit;

        assert(!areCodeActionsEqual(codeAction1, codeAction2));
    });

    it('should treat actions with different titles as inequal', () => {
        const codeAction1 = new vscode.CodeAction(
            'Fix me!',
            vscode.CodeActionKind.Refactor
        );

        const codeAction2 = new vscode.CodeAction(
            'Do something different!',
            vscode.CodeActionKind.Refactor
        );

        const edit = new vscode.WorkspaceEdit();
        edit.replace(uri, range1, 'Replace with this');
        codeAction1.edit = edit;
        codeAction2.edit = edit;

        assert(!areCodeActionsEqual(codeAction1, codeAction2));
    });

    it('should treat actions with different edits as inequal', () => {
        const codeAction1 = new vscode.CodeAction(
            'Fix me!',
            vscode.CodeActionKind.Refactor
        );
        const edit1 = new vscode.WorkspaceEdit();
        edit1.replace(uri, range1, 'Replace with this');
        codeAction1.edit = edit1;

        const codeAction2 = new vscode.CodeAction(
            'Fix me!',
            vscode.CodeActionKind.Refactor
        );
        const edit2 = new vscode.WorkspaceEdit();
        edit2.replace(uri, range1, 'Replace with this other thing');
        codeAction2.edit = edit2;

        assert(!areCodeActionsEqual(codeAction1, codeAction2));
    });
});
