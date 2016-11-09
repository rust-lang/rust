use serde::de;
use Value;
use super::{Decoder, DecodeError, DecodeErrorKind};
use std::collections::BTreeMap;

impl de::Deserializer for Decoder {
    type Error = DecodeError;

    fn deserialize<V>(&mut self, mut visitor: V)
                      -> Result<V::Value, DecodeError>
        where V: de::Visitor
    {
        match self.toml.take() {
            Some(Value::String(s)) => visitor.visit_string(s),
            Some(Value::Integer(i)) => visitor.visit_i64(i),
            Some(Value::Float(f)) => visitor.visit_f64(f),
            Some(Value::Boolean(b)) => visitor.visit_bool(b),
            Some(Value::Datetime(s)) => visitor.visit_string(s),
            Some(Value::Array(a)) => {
                let len = a.len();
                let iter = a.into_iter();
                visitor.visit_seq(SeqDeserializer::new(iter, len, &mut self.toml))
            }
            Some(Value::Table(t)) => {
                visitor.visit_map(MapVisitor {
                    iter: t.into_iter(),
                    de: self,
                    key: None,
                    value: None,
                })
            }
            None => Err(self.err(DecodeErrorKind::EndOfStream)),
        }
    }

    fn deserialize_bool<V>(&mut self, mut visitor: V)
                           -> Result<V::Value, DecodeError>
        where V: de::Visitor
    {
        match self.toml.take() {
            Some(Value::Boolean(b)) => visitor.visit_bool(b),
            ref found => Err(self.mismatch("bool", found)),
        }
    }

    fn deserialize_i64<V>(&mut self, mut visitor: V)
                          -> Result<V::Value, DecodeError>
        where V: de::Visitor
    {
        match self.toml.take() {
            Some(Value::Integer(f)) => visitor.visit_i64(f),
            ref found => Err(self.mismatch("integer", found)),
        }
    }

    fn deserialize_u64<V>(&mut self, v: V) -> Result<V::Value, DecodeError>
        where V: de::Visitor
    {
        self.deserialize_i64(v)
    }

    fn deserialize_f64<V>(&mut self, mut visitor: V)
                          -> Result<V::Value, DecodeError>
        where V: de::Visitor
    {
        match self.toml.take() {
            Some(Value::Float(f)) => visitor.visit_f64(f),
            ref found => Err(self.mismatch("float", found)),
        }
    }

    fn deserialize_str<V>(&mut self, mut visitor: V)
                          -> Result<V::Value, Self::Error>
        where V: de::Visitor,
    {
        match self.toml.take() {
            Some(Value::String(s)) => visitor.visit_string(s),
            ref found => Err(self.mismatch("string", found)),
        }
    }

    fn deserialize_char<V>(&mut self, mut visitor: V)
                           -> Result<V::Value, DecodeError>
        where V: de::Visitor
    {
        match self.toml.take() {
            Some(Value::String(ref s)) if s.chars().count() == 1 => {
                visitor.visit_char(s.chars().next().unwrap())
            }
            ref found => return Err(self.mismatch("string", found)),
        }
    }

    fn deserialize_option<V>(&mut self, mut visitor: V)
                             -> Result<V::Value, DecodeError>
        where V: de::Visitor
    {
        if self.toml.is_none() {
            visitor.visit_none()
        } else {
            visitor.visit_some(self)
        }
    }

    fn deserialize_seq<V>(&mut self, mut visitor: V)
                          -> Result<V::Value, DecodeError>
        where V: de::Visitor,
    {
        if self.toml.is_none() {
            let iter = None::<i32>.into_iter();
            visitor.visit_seq(de::value::SeqDeserializer::new(iter, 0))
        } else {
            self.deserialize(visitor)
        }
    }

    fn deserialize_map<V>(&mut self, mut visitor: V)
                          -> Result<V::Value, DecodeError>
        where V: de::Visitor,
    {
        match self.toml.take() {
            Some(Value::Table(t)) => {
                visitor.visit_map(MapVisitor {
                    iter: t.into_iter(),
                    de: self,
                    key: None,
                    value: None,
                })
            }
            ref found => Err(self.mismatch("table", found)),
        }
    }

    fn deserialize_enum<V>(&mut self,
                           _enum: &str,
                           variants: &[&str],
                           mut visitor: V) -> Result<V::Value, DecodeError>
        where V: de::EnumVisitor,
    {
        // When decoding enums, this crate takes the strategy of trying to
        // decode the current TOML as all of the possible variants, returning
        // success on the first one that succeeds.
        //
        // Note that fidelity of the errors returned here is a little nebulous,
        // but we try to return the error that had the relevant field as the
        // longest field. This way we hopefully match an error against what was
        // most likely being written down without losing too much info.
        let mut first_error = None::<DecodeError>;

        for variant in 0..variants.len() {
            let mut de = VariantVisitor {
                de: self.sub_decoder(self.toml.clone(), ""),
                variant: variant,
            };

            match visitor.visit(&mut de) {
                Ok(value) => {
                    self.toml = de.de.toml;
                    return Ok(value);
                }
                Err(e) => {
                    if let Some(ref first) = first_error {
                        let my_len = e.field.as_ref().map(|s| s.len());
                        let first_len = first.field.as_ref().map(|s| s.len());
                        if my_len <= first_len {
                            continue
                        }
                    }
                    first_error = Some(e);
                }
            }
        }

        Err(first_error.unwrap_or_else(|| self.err(DecodeErrorKind::NoEnumVariants)))
    }

    // When #[derive(Deserialize)] encounters an unknown struct field it will
    // call this method (somehow), and we want to preserve all unknown struct
    // fields to return them upwards (to warn about unused keys), so we override
    // that here to not tamper with our own internal state.
    fn deserialize_ignored_any<V>(&mut self, visitor: V)
                                  -> Result<V::Value, Self::Error>
        where V: de::Visitor
    {
        use serde::de::value::ValueDeserializer;
        let mut d = <() as ValueDeserializer<Self::Error>>::into_deserializer(());
        d.deserialize(visitor)
    }
}

struct VariantVisitor {
    de: Decoder,
    variant: usize,
}

impl de::VariantVisitor for VariantVisitor {
    type Error = DecodeError;

    fn visit_variant<V>(&mut self) -> Result<V, DecodeError>
        where V: de::Deserialize
    {
        use serde::de::value::ValueDeserializer;

        let mut de = self.variant.into_deserializer();

        de::Deserialize::deserialize(&mut de)
    }

    fn visit_unit(&mut self) -> Result<(), DecodeError> {
        de::Deserialize::deserialize(&mut self.de)
    }

    fn visit_newtype<T>(&mut self) -> Result<T, DecodeError>
        where T: de::Deserialize,
    {
        de::Deserialize::deserialize(&mut self.de)
    }

    fn visit_tuple<V>(&mut self,
                      _len: usize,
                      visitor: V) -> Result<V::Value, DecodeError>
        where V: de::Visitor,
    {
        de::Deserializer::deserialize(&mut self.de, visitor)
    }

    fn visit_struct<V>(&mut self,
                       _fields: &'static [&'static str],
                       visitor: V) -> Result<V::Value, DecodeError>
        where V: de::Visitor,
    {
        de::Deserializer::deserialize(&mut self.de, visitor)
    }
}

struct SeqDeserializer<'a, I> {
    iter: I,
    len: usize,
    toml: &'a mut Option<Value>,
}

impl<'a, I> SeqDeserializer<'a, I> where I: Iterator<Item=Value> {
    fn new(iter: I, len: usize, toml: &'a mut Option<Value>) -> Self {
        SeqDeserializer {
            iter: iter,
            len: len,
            toml: toml,
        }
    }

    fn put_value_back(&mut self, v: Value) {
        *self.toml = self.toml.take().or(Some(Value::Array(Vec::new())));
        match self.toml.as_mut().unwrap() {
            &mut Value::Array(ref mut a) => {
                a.push(v);
            },
            _ => unreachable!(),
        }
    }
}

impl<'a, I> de::Deserializer for SeqDeserializer<'a, I>
    where I: Iterator<Item=Value>,
{
    type Error = DecodeError;

    fn deserialize<V>(&mut self, mut visitor: V)
                      -> Result<V::Value, DecodeError>
        where V: de::Visitor,
    {
        visitor.visit_seq(self)
    }
}

impl<'a, I> de::SeqVisitor for SeqDeserializer<'a, I>
    where I: Iterator<Item=Value>
{
    type Error = DecodeError;

    fn visit<V>(&mut self) -> Result<Option<V>, DecodeError>
        where V: de::Deserialize
    {
        match self.iter.next() {
            Some(value) => {
                self.len -= 1;
                let mut de = Decoder::new(value);
                let v = try!(de::Deserialize::deserialize(&mut de));
                if let Some(t) = de.toml {
                    self.put_value_back(t);
                }
                Ok(Some(v))
            }
            None => Ok(None),
        }
    }

    fn end(&mut self) -> Result<(), DecodeError> {
        if self.len == 0 {
            Ok(())
        } else {
            Err(de::Error::end_of_stream())
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl de::Error for DecodeError {
    fn custom<T: Into<String>>(msg: T) -> DecodeError {
        DecodeError {
            field: None,
            kind: DecodeErrorKind::CustomError(msg.into()),
        }
    }
    fn end_of_stream() -> DecodeError {
        DecodeError { field: None, kind: DecodeErrorKind::EndOfStream }
    }
    fn missing_field(name: &'static str) -> DecodeError {
        DecodeError {
            field: Some(name.to_string()),
            kind: DecodeErrorKind::ExpectedField(None),
        }
    }
    fn unknown_field(name: &str) -> DecodeError {
        DecodeError {
            field: Some(name.to_string()),
            kind: DecodeErrorKind::UnknownField,
        }
    }
    fn invalid_type(ty: de::Type) -> Self {
        DecodeError {
            field: None,
            kind: DecodeErrorKind::InvalidType(match ty {
                de::Type::Bool => "bool",
                de::Type::Usize |
                de::Type::U8 |
                de::Type::U16 |
                de::Type::U32 |
                de::Type::U64 |
                de::Type::Isize |
                de::Type::I8 |
                de::Type::I16 |
                de::Type::I32 |
                de::Type::I64 => "integer",
                de::Type::F32 |
                de::Type::F64 => "float",
                de::Type::Char |
                de::Type::Str |
                de::Type::String => "string",
                de::Type::Seq => "array",
                de::Type::Struct |
                de::Type::Map => "table",
                de::Type::Unit => "Unit",
                de::Type::Option => "Option",
                de::Type::UnitStruct => "UnitStruct",
                de::Type::NewtypeStruct => "NewtypeStruct",
                de::Type::TupleStruct => "TupleStruct",
                de::Type::FieldName => "FieldName",
                de::Type::Tuple => "Tuple",
                de::Type::Enum => "Enum",
                de::Type::VariantName => "VariantName",
                de::Type::StructVariant => "StructVariant",
                de::Type::TupleVariant => "TupleVariant",
                de::Type::UnitVariant => "UnitVariant",
                de::Type::Bytes => "Bytes",
            })
        }
    }
}

struct MapVisitor<'a, I> {
    iter: I,
    de: &'a mut Decoder,
    key: Option<String>,
    value: Option<Value>,
}

impl<'a, I> MapVisitor<'a, I> {
    fn put_value_back(&mut self, v: Value) {
        self.de.toml = self.de.toml.take().or_else(|| {
            Some(Value::Table(BTreeMap::new()))
        });

        match self.de.toml.as_mut().unwrap() {
            &mut Value::Table(ref mut t) => {
                t.insert(self.key.take().unwrap(), v);
            },
            _ => unreachable!(),
        }
    }
}

impl<'a, I> de::MapVisitor for MapVisitor<'a, I>
    where I: Iterator<Item=(String, Value)>
{
    type Error = DecodeError;

    fn visit_key<K>(&mut self) -> Result<Option<K>, DecodeError>
        where K: de::Deserialize
    {
        while let Some((k, v)) = self.iter.next() {
            let mut dec = self.de.sub_decoder(Some(Value::String(k.clone())), &k);
            self.key = Some(k);

            match de::Deserialize::deserialize(&mut dec) {
                Ok(val) => {
                    self.value = Some(v);
                    return Ok(Some(val))
                }

                // If this was an unknown field, then we put the toml value
                // back into the map and keep going.
                Err(DecodeError {kind: DecodeErrorKind::UnknownField, ..}) => {
                    self.put_value_back(v);
                }

                Err(e) => return Err(e),
            }
        }
        Ok(None)
    }

    fn visit_value<V>(&mut self) -> Result<V, DecodeError>
        where V: de::Deserialize
    {
        match self.value.take() {
            Some(t) => {
                let mut dec = {
                    // Borrowing the key here because Rust doesn't have
                    // non-lexical borrows yet.
                    let key = match self.key {
                        Some(ref key) => &**key,
                        None => ""
                    };

                    self.de.sub_decoder(Some(t), key)
                };
                let v = try!(de::Deserialize::deserialize(&mut dec));
                if let Some(t) = dec.toml {
                    self.put_value_back(t);
                }
                Ok(v)
            },
            None => Err(de::Error::end_of_stream())
        }
    }

    fn end(&mut self) -> Result<(), DecodeError> {
        if let Some(v) = self.value.take() {
            self.put_value_back(v);
        }
        while let Some((k, v)) = self.iter.next() {
            self.key = Some(k);
            self.put_value_back(v);
        }
        Ok(())
    }

    fn missing_field<V>(&mut self, field_name: &'static str)
                        -> Result<V, DecodeError> where V: de::Deserialize {
        // See if the type can deserialize from a unit.
        match de::Deserialize::deserialize(&mut UnitDeserializer) {
            Err(DecodeError {
                kind: DecodeErrorKind::InvalidType(..),
                field,
            }) => Err(DecodeError {
                field: field.or(Some(field_name.to_string())),
                kind: DecodeErrorKind::ExpectedField(None),
            }),
            v => v,
        }
    }
}

struct UnitDeserializer;

impl de::Deserializer for UnitDeserializer {
    type Error = DecodeError;

    fn deserialize<V>(&mut self, mut visitor: V)
                      -> Result<V::Value, DecodeError>
        where V: de::Visitor,
    {
        visitor.visit_unit()
    }

    fn deserialize_option<V>(&mut self, mut visitor: V)
                             -> Result<V::Value, DecodeError>
        where V: de::Visitor,
    {
        visitor.visit_none()
    }
}

impl de::Deserialize for Value {
    fn deserialize<D>(deserializer: &mut D) -> Result<Value, D::Error>
        where D: de::Deserializer
    {
        struct ValueVisitor;

        impl de::Visitor for ValueVisitor {
            type Value = Value;

            fn visit_bool<E>(&mut self, value: bool) -> Result<Value, E> {
                Ok(Value::Boolean(value))
            }

            fn visit_i64<E>(&mut self, value: i64) -> Result<Value, E> {
                Ok(Value::Integer(value))
            }

            fn visit_f64<E>(&mut self, value: f64) -> Result<Value, E> {
                Ok(Value::Float(value))
            }

            fn visit_str<E>(&mut self, value: &str) -> Result<Value, E> {
                Ok(Value::String(value.into()))
            }

            fn visit_string<E>(&mut self, value: String) -> Result<Value, E> {
                Ok(Value::String(value))
            }

            fn visit_seq<V>(&mut self, visitor: V) -> Result<Value, V::Error>
                where V: de::SeqVisitor
            {
                let values = try!(de::impls::VecVisitor::new().visit_seq(visitor));
                Ok(Value::Array(values))
            }

            fn visit_map<V>(&mut self, visitor: V) -> Result<Value, V::Error>
                where V: de::MapVisitor
            {
                let mut v = de::impls::BTreeMapVisitor::new();
                let values = try!(v.visit_map(visitor));
                Ok(Value::Table(values))
            }
        }

        deserializer.deserialize(ValueVisitor)
    }
}
