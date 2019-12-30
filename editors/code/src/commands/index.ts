import { Ctx, Cmd } from '../ctx'

import { analyzerStatus } from './analyzer_status';
import { matchingBrace } from './matching_brace';
import { joinLines } from './join_lines';
import * as applySourceChange from './apply_source_change';
import * as expandMacro from './expand_macro';
import * as inlayHints from './inlay_hints';
import * as onEnter from './on_enter';
import * as parentModule from './parent_module';
import * as runnables from './runnables';
import * as syntaxTree from './syntaxTree';

function collectGarbage(ctx: Ctx): Cmd {
    return async () => { ctx.client.sendRequest<null>('rust-analyzer/collectGarbage', null) }
}

export {
    analyzerStatus,
    applySourceChange,
    expandMacro,
    joinLines,
    matchingBrace,
    parentModule,
    runnables,
    syntaxTree,
    onEnter,
    inlayHints,
    collectGarbage
};
