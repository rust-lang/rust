import * as assert from 'assert';
import * as vscode from 'vscode';

import { areDiagnosticsEqual } from '../../../utils/diagnostics/vscode';

const range1 = new vscode.Range(
    new vscode.Position(1, 2),
    new vscode.Position(3, 4),
);

const range2 = new vscode.Range(
    new vscode.Position(5, 6),
    new vscode.Position(7, 8),
);

describe('areDiagnosticsEqual', () => {
    it('should treat identical diagnostics as equal', () => {
        const diagnostic1 = new vscode.Diagnostic(
            range1,
            'Hello, world!',
            vscode.DiagnosticSeverity.Error,
        );

        const diagnostic2 = new vscode.Diagnostic(
            range1,
            'Hello, world!',
            vscode.DiagnosticSeverity.Error,
        );

        assert(areDiagnosticsEqual(diagnostic1, diagnostic2));
    });

    it('should treat diagnostics with different sources as inequal', () => {
        const diagnostic1 = new vscode.Diagnostic(
            range1,
            'Hello, world!',
            vscode.DiagnosticSeverity.Error,
        );
        diagnostic1.source = 'rustc';

        const diagnostic2 = new vscode.Diagnostic(
            range1,
            'Hello, world!',
            vscode.DiagnosticSeverity.Error,
        );
        diagnostic2.source = 'clippy';

        assert(!areDiagnosticsEqual(diagnostic1, diagnostic2));
    });

    it('should treat diagnostics with different ranges as inequal', () => {
        const diagnostic1 = new vscode.Diagnostic(
            range1,
            'Hello, world!',
            vscode.DiagnosticSeverity.Error,
        );

        const diagnostic2 = new vscode.Diagnostic(
            range2,
            'Hello, world!',
            vscode.DiagnosticSeverity.Error,
        );

        assert(!areDiagnosticsEqual(diagnostic1, diagnostic2));
    });

    it('should treat diagnostics with different messages as inequal', () => {
        const diagnostic1 = new vscode.Diagnostic(
            range1,
            'Hello, world!',
            vscode.DiagnosticSeverity.Error,
        );

        const diagnostic2 = new vscode.Diagnostic(
            range1,
            'Goodbye!, world!',
            vscode.DiagnosticSeverity.Error,
        );

        assert(!areDiagnosticsEqual(diagnostic1, diagnostic2));
    });

    it('should treat diagnostics with different severities as inequal', () => {
        const diagnostic1 = new vscode.Diagnostic(
            range1,
            'Hello, world!',
            vscode.DiagnosticSeverity.Warning,
        );

        const diagnostic2 = new vscode.Diagnostic(
            range1,
            'Hello, world!',
            vscode.DiagnosticSeverity.Error,
        );

        assert(!areDiagnosticsEqual(diagnostic1, diagnostic2));
    });
});
