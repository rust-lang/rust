//This test used to ICE: #117629
//@ edition:2021

#![feature(const_trait_impl)]

#[const_trait]
trait Tr {
    (const) async fn ft1() {}
    //~^ ERROR: functions cannot be both `const` and `async`
    async fn ft2() {}
    //~^ ERROR: non-const fn in const traits are not supported yet
    const async fn ft3() {}
    //~^ ERROR: functions in traits cannot be declared const
    //~| ERROR: functions cannot be both `const` and `async`
    async const fn ft4() {}
    //~^ ERROR: expected one of `extern`, `fn`, `safe`, or `unsafe`, found keyword `const`
    //~| ERROR: functions in traits cannot be declared const
    //~| ERROR: functions cannot be both `const` and `async`
    async (const) fn ft5() {}
    //~^ ERROR: non-item in item list
}

fn main() {}
