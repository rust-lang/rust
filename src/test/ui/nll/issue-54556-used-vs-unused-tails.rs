// Ths test case is exploring the space of how blocs with tail
// expressions and statements can be composed, trying to keep each
// case on one line so that we can compare them via a vertical scan
// with the human eye.

// Each comment on the right side of the line is summarizing the
// expected suggestion from the diagnostic for issue #54556.

fn main() {
    {              let mut _t1 = D(Box::new("t1")); D(&_t1).end()    } ; // suggest `;`
//~^ ERROR does not live long enough

    {            { let mut _t1 = D(Box::new("t1")); D(&_t1).end() }  } ; // suggest `;`
//~^ ERROR does not live long enough

    {            { let mut _t1 = D(Box::new("t1")); D(&_t1).end() }; }   // suggest `;`
//~^ ERROR does not live long enough

    let _ =      { let mut _t1 = D(Box::new("t1")); D(&_t1).end()    } ; // suggest `;`
//~^ ERROR does not live long enough

    let _u =     { let mut _t1 = D(Box::new("t1")); D(&_t1).unit()   } ; // suggest `;`
//~^ ERROR does not live long enough

    let _x =     { let mut _t1 = D(Box::new("t1")); D(&_t1).end()    } ; // `let x = ...; x`
//~^ ERROR does not live long enough
    let _x =     { let mut _t1 = D(Box::new("t1")); let x = D(&_t1).end(); x } ; // no error

    let mut _y;
    _y =         { let mut _t1 = D(Box::new("t1")); D(&_t1).end() } ; // `let x = ...; x`
//~^ ERROR does not live long enough
    _y =         { let mut _t1 = D(Box::new("t1")); let x = D(&_t1).end(); x } ; // no error
}

fn f_param_ref(_t1: D<Box<&'static str>>) {         D(&_t1).unit()   }  // no error

fn f_local_ref() { let mut _t1 = D(Box::new("t1")); D(&_t1).unit()   }  // suggest `;`
//~^ ERROR does not live long enough

fn f() -> String { let mut _t1 = D(Box::new("t1")); D(&_t1).end()   }   // `let x = ...; x`
//~^ ERROR does not live long enough

#[derive(Debug)]
struct D<T: std::fmt::Debug>(T);

impl<T: std::fmt::Debug>  Drop for D<T> {
    fn drop(&mut self) {
        println!("dropping {:?})", self);
    }
}

impl<T: std::fmt::Debug> D<T> {
    fn next<U: std::fmt::Debug>(&self, _other: U) -> D<U> { D(_other) }
    fn end(&self) -> String { format!("End({:?})", self.0) }
    fn unit(&self) { }
}
