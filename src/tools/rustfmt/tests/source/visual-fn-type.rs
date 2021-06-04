// rustfmt-indent_style: Visual
type CNodeSetAtts = unsafe extern "C" fn(node: *const RsvgNode,
                                         node_impl: *const RsvgCNodeImpl,
                                         handle: *const RsvgHandle,
                                         pbag: *const PropertyBag)
                                         ;
type CNodeDraw = unsafe extern "C" fn(node: *const RsvgNode,
                                      node_impl: *const RsvgCNodeImpl,
                                      draw_ctx: *const RsvgDrawingCtx,
                                      dominate: i32);
