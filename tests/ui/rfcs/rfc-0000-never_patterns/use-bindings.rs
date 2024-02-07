#![feature(never_patterns)]
#![allow(incomplete_features)]

#[derive(Copy, Clone)]
enum Void {}

fn main() {
    let res_void: Result<bool, Void> = Ok(true);

    let (Ok(x) | Err(!)) = res_void;
    println!("{x}");
    //~^ ERROR: used binding `x` is possibly-uninitialized
    let (Ok(x) | Err(!)) = &res_void;
    println!("{x}");
    //~^ ERROR: used binding `x` is possibly-uninitialized
    let (Err(!) | Ok(x)) = res_void;
    println!("{x}");
    //~^ ERROR: used binding `x` is possibly-uninitialized

    match res_void {
        Ok(x) | Err(!) => println!("{x}"),
        //~^ ERROR: used binding `x` is possibly-uninitialized
    }
    match res_void {
        Err(!) | Ok(x) => println!("{x}"),
        //~^ ERROR: used binding `x` is possibly-uninitialized
    }

    let res_res_void: Result<Result<bool, Void>, Void> = Ok(Ok(true));
    match res_res_void {
        Ok(Ok(x) | Err(!)) | Err(!) => println!("{x}"),
        //~^ ERROR: used binding `x` is possibly-uninitialized
    }
}
