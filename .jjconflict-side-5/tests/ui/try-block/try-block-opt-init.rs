//@ edition: 2018

#![feature(try_blocks)]

fn use_val<T: Sized>(_x: T) {}

pub fn main() {
    let cfg_res;
    let _: Result<(), ()> = try {
        Err(())?;
        cfg_res = 5;
        Ok::<(), ()>(())?;
        use_val(cfg_res);
    };
    assert_eq!(cfg_res, 5); //~ ERROR E0381
}
