#![forbid(deprecated)] //~ NOTE `forbid` level set here

#[allow(deprecated)]
//~^ ERROR allow(deprecated) incompatible with previous forbid [E0453]
//~^^ NOTE overruled by previous forbid
fn main() {
}
