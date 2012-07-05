
import codemap::span;
import ext::base::ext_ctxt;

import pipes::pipec::*;

fn expand_proto(cx: ext_ctxt, span: span, id: ast::ident, tt: ast::token_tree)
    -> @ast::item
{
    let proto = protocol(id);
    let ping = proto.add_state(@"ping", send);
    let pong = proto.add_state(@"pong", recv);

    ping.add_message(@"ping", []/~, pong, ~[]);
    pong.add_message(@"pong", []/~, ping, ~[]);
    proto.compile(cx)
}