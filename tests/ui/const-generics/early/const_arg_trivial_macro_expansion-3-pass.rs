// Additional checks for macro expansion in const args

//@ check-pass

macro_rules! closure {
    () => { |()| () };
}

macro_rules! indir_semi {
    ($nested:ident) => { $nested!(); };
}

macro_rules! indir {
    ($nested:ident) => { $nested!() };
}

macro_rules! empty {
    () => {};
}

macro_rules! arg {
    () => { N };
}

struct Adt<const N: usize>;

fn array1() -> [(); { closure!(); 0 }] { loop {} }
fn array2() -> [(); { indir!(closure); 0}] { loop {} }
fn array3() -> [(); { indir_semi!{ closure } 0 }] { loop {} }
fn array4<const N: usize>() -> [(); { indir!{ empty } arg!{} }] { loop {} }
fn array5<const N: usize>() -> [(); { empty!{} arg!() }] { loop {} }
fn array6<const N: usize>() -> [(); { empty!{} N }] { loop {} }
fn array7<const N: usize>() -> [(); { arg!{} empty!{} }] { loop {} }
fn array8<const N: usize>() -> [(); { empty!{} arg!{} empty!{} }] { loop {} }

fn adt1() -> Adt<{ closure!(); 0 }> { loop {} }
fn adt2() -> Adt<{ indir!(closure); 0}> { loop {} }
fn adt3() -> Adt<{ indir_semi!{ closure } 0 }> { loop {} }
fn adt4<const N: usize>() -> Adt<{ indir!{ empty } arg!{} }> { loop {} }
fn adt5<const N: usize>() -> Adt<{ empty!{} arg!() }> { loop {} }
fn adt6<const N: usize>() -> Adt<{ empty!{} N }> { loop {} }
fn adt7<const N: usize>() -> Adt<{ arg!{} empty!{} }> { loop {} }
fn adt8<const N: usize>() -> Adt<{ empty!{} arg!{} empty!{} }> { loop {} }


fn main() {}
