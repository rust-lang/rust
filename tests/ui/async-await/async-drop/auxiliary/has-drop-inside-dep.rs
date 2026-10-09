//@ edition: 2024

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
