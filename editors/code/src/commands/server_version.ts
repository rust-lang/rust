import * as vscode from 'vscode';
import { ServerVersion } from '../installation/server';
import { Cmd } from '../ctx';

export function serverVersion(): Cmd {
    return () => {
        vscode.window.showInformationMessage('rust-analyzer version : ' + ServerVersion);
    };
}

