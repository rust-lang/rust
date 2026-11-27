// This is the auxiliary crate for the regression test for issue #89119, minimized
// from `zvariant-2.8.0`.

use std::convert::TryFrom;
use std::borrow::Cow;

pub struct Str<'a>(Cow<'a, str>);
impl<'a> Str<'a> {
    pub fn to_owned(&self) -> Str<'static> {
        todo!()
    }
}

pub enum Value<'a> {
    Str(Str<'a>),
    Value(Box<Value<'a>>),
}
impl<'a> Value<'a> {
    pub fn to_owned(&self) -> Value<'static> {
        match self {
            Value::Str(v) => Value::Str(v.to_owned()),
            Value::Value(v) => {
                let o = OwnedValue::from(&**v);
                Value::Value(Box::new(o.into_inner()))
            }
        }
    }
}

struct OwnedValue(Value<'static>);
impl OwnedValue {
    pub(crate) fn into_inner(self) -> Value<'static> {
        todo!()
    }
}
impl<'a, T> TryFrom<OwnedValue> for Vec<T>
where
    T: TryFrom<Value<'a>, Error = ()>,
{
    type Error = ();
    fn try_from(_: OwnedValue) -> Result<Self, Self::Error> {
        todo!()
    }
}
impl TryFrom<OwnedValue> for Vec<OwnedValue> {
    type Error = ();
    fn try_from(_: OwnedValue) -> Result<Self, Self::Error> {
        todo!()
    }
}
impl<'a> From<Value<'a>> for OwnedValue {
    fn from(_: Value<'a>) -> Self {
        todo!()
    }
}
impl<'a> From<&Value<'a>> for OwnedValue {
    fn from(_: &Value<'a>) -> Self {
        todo!()
    }
}
