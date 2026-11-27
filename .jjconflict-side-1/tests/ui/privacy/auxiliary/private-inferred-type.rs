#![feature(decl_macro)]

fn priv_fn() {}
static PRIV_STATIC: u8 = 0;
enum PrivEnum { Variant }
pub enum PubEnum { Variant }
trait PrivTrait { fn method() {} }
impl PrivTrait for u8 {}
pub trait PubTrait { fn method() {} }
impl PubTrait for u8 {}
struct PrivTupleStruct(u8);
pub struct PubTupleStruct(u8);
impl PubTupleStruct { fn method() {} }

struct Priv;
pub type Alias = Priv;
pub struct Pub<T = Alias>(pub T);

impl Pub<Priv> {
    pub fn static_method() {}
}
impl Pub<u8> {
    fn priv_method(&self) {}
}

pub macro m() {
    priv_fn;
    PRIV_STATIC;
    PrivEnum::Variant;
    PubEnum::Variant;
    <u8 as PrivTrait>::method;
    <u8 as PubTrait>::method;
    PrivTupleStruct;
    PubTupleStruct;
    Pub(0u8).priv_method();
}
