#[macro_use]
extern crate neon;
extern crate libeditor;

use neon::prelude::*;
use libeditor::TextRange;

pub struct Wrapper {
    inner: libeditor::File,
}

declare_types! {
    /// A class for generating greeting strings.
    pub class RustFile for Wrapper {
        init(mut cx) {
            let text = cx.argument::<JsString>(0)?.value();
            Ok(Wrapper {
                inner: libeditor::File::new(&text)
            })
        }

        method syntaxTree(mut cx) {
            let tree = {
            let this = cx.this();
                let guard = cx.lock();
                let wrapper = this.borrow(&guard);
                wrapper.inner.syntax_tree()
            };
            Ok(cx.string(tree.as_str()).upcast())
        }

        method highlight(mut cx) {
            let highlights = {
                let this = cx.this();
                let guard = cx.lock();
                let wrapper = this.borrow(&guard);
                wrapper.inner.highlight()
            };
            let res = cx.empty_array();
            for (i, hl) in highlights.into_iter().enumerate() {
                let start: u32 = hl.range.start().into();
                let end: u32 = hl.range.end().into();
                let start = cx.number(start);
                let end = cx.number(end);
                let tag = cx.string(hl.tag);
                let hl = cx.empty_array();
                hl.set(&mut cx, 0, start)?;
                hl.set(&mut cx, 1, end)?;
                hl.set(&mut cx, 2, tag)?;
                res.set(&mut cx, i as u32, hl)?;
            }

            Ok(res.upcast())
        }

        method extendSelection(mut cx) {
            let from_offset = cx.argument::<JsNumber>(0)?.value() as u32;
            let to_offset = cx.argument::<JsNumber>(1)?.value() as u32;
            let text_range = TextRange::from_to(from_offset.into(), to_offset.into());
            let extended_range = {
                let this = cx.this();
                let guard = cx.lock();
                let wrapper = this.borrow(&guard);
                wrapper.inner.extend_selection(text_range)
            };

            match extended_range {
                None => Ok(cx.null().upcast()),
                Some(range) => {
                    let start: u32 = range.start().into();
                    let end: u32 = range.end().into();
                    let start = cx.number(start);
                    let end = cx.number(end);
                    let arr = cx.empty_array();
                    arr.set(&mut cx, 0, start)?;
                    arr.set(&mut cx, 1, end)?;
                    Ok(arr.upcast())
                }
            }

        }
    }

}

register_module!(mut cx, {
    cx.export_class::<RustFile>("RustFile")
});
