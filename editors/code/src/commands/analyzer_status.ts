import * as vscode from 'vscode';
import { Server } from '../server';

// Shows status of rust-analyzer (for debugging)
export async function handle() {
    const status = await Server.client.sendRequest<string>(
        'ra/analyzerStatus',
        null
    );
    const doc = await vscode.workspace.openTextDocument({ content: status });
    await vscode.window.showTextDocument(doc, vscode.ViewColumn.Two);
}
