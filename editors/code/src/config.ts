import * as vscode from 'vscode';

import { Server } from './server';

export class Config {
  public highlightingOn = true;

  constructor() {
    vscode.workspace.onDidChangeConfiguration((_) => this.userConfigChanged());
    this.userConfigChanged();
  }

  public userConfigChanged() {
    const config = vscode.workspace.getConfiguration('ra-lsp');
    if (config.has('highlightingOn')) {
      this.highlightingOn = config.get('highlightingOn') as boolean;
    }

    if (!this.highlightingOn && Server) {
      Server.highlighter.removeHighlights();
    }
  }
}
