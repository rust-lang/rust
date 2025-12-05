//@ ignore-backends: gcc
//@ build-pass
//@ compile-flags: -Copt-level=s -Clto=fat
//@ no-prefer-dynamic
//@ edition: 2021

#![recursion_limit = "256"]

fn main() {
    spawn(move || main0())
}

fn spawn<F>(future: impl FnOnce() -> F) {
    future();
}

async fn main0() {
    main1().await;
    main2().await;
}
async fn main1() {
    main2().await;
    main3().await;
}
async fn main2() {
    main3().await;
    main4().await;
}
async fn main3() {
    main4().await;
    main5().await;
}
async fn main4() {
    main5().await;
    main6().await;
}
async fn main5() {
    main6().await;
    main7().await;
}
async fn main6() {
    main7().await;
    main8().await;
}
async fn main7() {
    main8().await;
    main9().await;
}
async fn main8() {
    main9().await;
    main10().await;
}
async fn main9() {
    main10().await;
    main11().await;
}
async fn main10() {
    main11().await;
    main12().await;
}
async fn main11() {
    main12().await;
    main13().await;
}
async fn main12() {
    main13().await;
    main14().await;
}
async fn main13() {
    main14().await;
    main15().await;
}
async fn main14() {
    main15().await;
    main16().await;
}
async fn main15() {
    main16().await;
    main17().await;
}
async fn main16() {
    main17().await;
    main18().await;
}
async fn main17() {
    main18().await;
    main19().await;
}
async fn main18() {
    main19().await;
    main20().await;
}
async fn main19() {
    main20().await;
    main21().await;
}
async fn main20() {
    main21().await;
    main22().await;
}
async fn main21() {
    main22().await;
    main23().await;
}
async fn main22() {
    main23().await;
    main24().await;
}
async fn main23() {
    main24().await;
    main25().await;
}
async fn main24() {
    main25().await;
    main26().await;
}
async fn main25() {
    main26().await;
    main27().await;
}
async fn main26() {
    main27().await;
    main28().await;
}
async fn main27() {
    main28().await;
    main29().await;
}
async fn main28() {
    main29().await;
    main30().await;
}
async fn main29() {
    main30().await;
    main31().await;
}
async fn main30() {
    main31().await;
    main32().await;
}
async fn main31() {
    main32().await;
    main33().await;
}
async fn main32() {
    main33().await;
    main34().await;
}
async fn main33() {
    main34().await;
    main35().await;
}
async fn main34() {
    main35().await;
    main36().await;
}
async fn main35() {
    main36().await;
    main37().await;
}
async fn main36() {
    main37().await;
    main38().await;
}
async fn main37() {
    main38().await;
    main39().await;
}
async fn main38() {
    main39().await;
    main40().await;
}
async fn main39() {
    main40().await;
}
async fn main40() {
    boom(&mut ()).await;
}

async fn boom(f: &mut ()) {}
