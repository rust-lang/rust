import * as vscode from 'vscode';
import { log } from './util';

export class PersistentState {
    constructor(private readonly globalState: vscode.Memento) {
    }
}
