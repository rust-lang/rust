import * as vscode from 'vscode';
import * as lc from 'vscode-languageclient';

import { Ctx, Cmd } from '../ctx';

import { analyzerStatus } from './analyzer_status';
import { matchingBrace } from './matching_brace';
import { joinLines } from './join_lines';
import { onEnter } from './on_enter';
import { parentModule } from './parent_module';
import { syntaxTree } from './syntax_tree';
import { expandMacro } from './expand_macro';
import { run, runSingle } from './runnables';

function collectGarbage(ctx: Ctx): Cmd {
    return async () => {
        ctx.client.sendRequest<null>('rust-analyzer/collectGarbage', null);
    };
}

function showReferences(ctx: Ctx): Cmd {
    return (uri: string, position: lc.Position, locations: lc.Location[]) => {
        vscode.commands.executeCommand(
            'editor.action.showReferences',
            vscode.Uri.parse(uri),
            ctx.client.protocol2CodeConverter.asPosition(position),
            locations.map(ctx.client.protocol2CodeConverter.asLocation),
        );
    };
}

export {
    analyzerStatus,
    expandMacro,
    joinLines,
    matchingBrace,
    parentModule,
    syntaxTree,
    onEnter,
    collectGarbage,
    run,
    runSingle,
    showReferences,
};
