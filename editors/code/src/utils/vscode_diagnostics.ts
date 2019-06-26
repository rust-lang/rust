import * as vscode from 'vscode';

/** Compares two `vscode.Diagnostic`s for equality */
export function areDiagnosticsEqual(
    left: vscode.Diagnostic,
    right: vscode.Diagnostic
): boolean {
    return (
        left.source === right.source &&
        left.severity === right.severity &&
        left.range.isEqual(right.range) &&
        left.message === right.message
    );
}

/** Compares two `vscode.TextEdit`s for equality */
function areTextEditsEqual(
    left: vscode.TextEdit,
    right: vscode.TextEdit
): boolean {
    if (!left.range.isEqual(right.range)) {
        return false;
    }

    if (left.newText !== right.newText) {
        return false;
    }

    return true;
}

/** Compares two `vscode.CodeAction`s for equality */
export function areCodeActionsEqual(
    left: vscode.CodeAction,
    right: vscode.CodeAction
): boolean {
    if (
        left.kind !== right.kind ||
        left.title !== right.title ||
        !left.edit ||
        !right.edit
    ) {
        return false;
    }

    const leftEditEntries = left.edit.entries();
    const rightEditEntries = right.edit.entries();

    if (leftEditEntries.length !== rightEditEntries.length) {
        return false;
    }

    for (let i = 0; i < leftEditEntries.length; i++) {
        const [leftUri, leftEdits] = leftEditEntries[i];
        const [rightUri, rightEdits] = rightEditEntries[i];

        if (leftUri.toString() !== rightUri.toString()) {
            return false;
        }

        if (leftEdits.length !== rightEdits.length) {
            return false;
        }

        for (let j = 0; j < leftEdits.length; j++) {
            if (!areTextEditsEqual(leftEdits[j], rightEdits[j])) {
                return false;
            }
        }
    }

    return true;
}
