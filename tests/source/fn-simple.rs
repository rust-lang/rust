
fn simple(/*pre-comment on a function!?*/ i: i32/*yes, it's possible!  */   
                                        ,response: NoWay /* hose */) {
fn op(x: Typ, key : &[u8], upd : Box<Fn(Option<&memcache::Item>) -> (memcache::Status, Result<memcache::Item, Option<String>>)>) -> MapResult {}

        "cool"}


fn weird_comment(/* /*/ double level */ comment */ x: Hello /*/*/* tripple, even */*/*/,
// Does this work?
y: World
) {
        simple(/* does this preserve comments now? */ 42, NoWay)
}

fn generic<T>(arg: T) -> &SomeType
    where T: Fn(// First arg
        A,
        // Second argument
        B, C, D, /* pre comment */ E /* last comment */) -> &SomeType {
    arg(a, b, c, d, e)    
}

fn foo()  ->  !  {}

pub fn http_fetch_async(listener:Box< AsyncCORSResponseListener+Send >,  script_chan:  Box<ScriptChan+Send>) {
}

fn some_func<T:Box<Trait+Bound>>(val:T){}

fn zzzzzzzzzzzzzzzzzzzz<Type, NodeType>
                       (selff: Type, mut handle: node::Handle<IdRef<'id, Node<K, V>>, Type, NodeType>)
                        -> SearchStack<'a, K, V, Type, NodeType>{
}

unsafe fn generic_call(cx: *mut JSContext, argc: libc::c_uint, vp: *mut JSVal,
    is_lenient: bool,
                       call: unsafe extern fn(*const JSJitInfo, *mut JSContext,
                                              HandleObject, *mut libc::c_void, u32,
                                              *mut JSVal)
                                              -> u8) {
    let f:  fn  ( _ , _  ) ->  _   =  panic!()  ;
}
