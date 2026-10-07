//@ edition: 2024

// WARNING: If you would ever want to modify this test,
// please consider modifying rustc's async drop test at
// `tests/ui/async-await/async-drop/auxiliary/has-drop-inside-dep.rs`.

#![feature(async_drop)]
#![allow(incomplete_features)]

pub struct IsDrop;
impl Drop for IsDrop {
    fn drop(&mut self) {
        // this is stub. no-op.
    }
}
pub struct HasDrop {
    _is_drop: IsDrop,
}

pub struct HasHasDrop {
    _has_drop: HasDrop,
}

pub async fn with_has_drop() {
    let _has_drop = HasDrop { _is_drop: IsDrop };
}

pub async fn with_has_has_drop() {
    let _has_has_drop = HasHasDrop { _has_drop: HasDrop { _is_drop: IsDrop } };
}
