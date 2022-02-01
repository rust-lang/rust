#![deny(suspicious_auto_trait_impls)]

struct MayImplementSendOk<T>(T);
unsafe impl<T: Send> Send for MayImplementSendOk<T> {} // ok

struct MayImplementSendErr<T>(T);
unsafe impl<T: Send> Send for MayImplementSendErr<&T> {}
//~^ ERROR
//~| WARNING this will change its meaning

struct ContainsNonSendDirect<T>(*const T);
unsafe impl<T: Send> Send for ContainsNonSendDirect<&T> {} // ok

struct ContainsPtr<T>(*const T);
struct ContainsIndirectNonSend<T>(ContainsPtr<T>);
unsafe impl<T: Send> Send for ContainsIndirectNonSend<&T> {} // ok

struct ContainsVec<T>(Vec<T>);
unsafe impl Send for ContainsVec<i32> {}
//~^ ERROR
//~| WARNING this will change its meaning

struct TwoParams<T, U>(T, U);
unsafe impl<T: Send, U: Send> Send for TwoParams<T, U> {} // ok

struct TwoParamsFlipped<T, U>(T, U);
unsafe impl<T: Send, U: Send> Send for TwoParamsFlipped<U, T> {} // ok

struct TwoParamsSame<T, U>(T, U);
unsafe impl<T: Send> Send for TwoParamsSame<T, T> {}
//~^ ERROR
//~| WARNING this will change its meaning

fn main() {}
