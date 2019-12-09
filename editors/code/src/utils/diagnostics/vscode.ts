import * as vscode from 'vscode';

/** Compares two `vscode.Diagnostic`s for equality */
export function areDiagnosticsEqual(
    left: vscode.Diagnostic,
    right: vscode.Diagnostic,
): boolean {
    return (
        left.source === right.source &&
        left.severity === right.severity &&
        left.range.isEqual(right.range) &&
        left.message === right.message
    );
}
