
fn main() {
    // FIXME(#133): the list rewrite should fail and force a different format
   let constellation_chan = Constellation::<layout::layout_task::LayoutTask,  script::script_task::ScriptTask> ::start(
     compositor_proxy,
     resource_task,
     image_cache_task,font_cache_task,
     time_profiler_chan,
     mem_profiler_chan,
     devtools_chan,
     storage_task,
     supports_clipboard
    );

     Quux::<ParamOne,   // Comment 1
            ParamTwo,   // Comment 2
                    >::some_func();
}

fn op(foo: Bar, key : &[u8], upd : Fn(Option<&memcache::Item> , Baz  ) -> Result) -> MapResult {}
