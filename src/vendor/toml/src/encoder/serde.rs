use serde::ser;
use Value;
use super::{Encoder, Error};

impl ser::Serializer for Encoder {
    type Error = Error;

    fn serialize_bool(&mut self, v: bool) -> Result<(), Error> {
        self.emit_value(Value::Boolean(v))
    }
    fn serialize_i64(&mut self, v: i64) -> Result<(), Error> {
        self.emit_value(Value::Integer(v))
    }
    fn serialize_u64(&mut self, v: u64) -> Result<(), Error> {
        self.serialize_i64(v as i64)
    }
    fn serialize_f64(&mut self, v: f64) -> Result<(), Error> {
        self.emit_value(Value::Float(v))
    }
    fn serialize_str(&mut self, value: &str) -> Result<(), Error> {
        self.emit_value(Value::String(value.to_string()))
    }
    fn serialize_unit(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn serialize_none(&mut self) -> Result<(), Error> {
        self.emit_none()
    }
    fn serialize_some<V>(&mut self, value: V) -> Result<(), Error>
        where V: ser::Serialize
    {
        value.serialize(self)
    }
    fn serialize_seq<V>(&mut self, mut visitor: V) -> Result<(), Error>
        where V: ser::SeqVisitor
    {
        self.seq(|me| {
            while try!(visitor.visit(me)).is_some() {}
            Ok(())
        })
    }
    fn serialize_seq_elt<T>(&mut self, value: T) -> Result<(), Error>
        where T: ser::Serialize
    {
        value.serialize(self)
    }
    fn serialize_map<V>(&mut self, mut visitor: V) -> Result<(), Error>
        where V: ser::MapVisitor
    {
        self.table(|me| {
            while try!(visitor.visit(me)).is_some() {}
            Ok(())
        })
    }
    fn serialize_map_elt<K, V>(&mut self, key: K, value: V) -> Result<(), Error>
        where K: ser::Serialize, V: ser::Serialize
    {
        try!(self.table_key(|me| key.serialize(me)));
        try!(value.serialize(self));
        Ok(())
    }
    fn serialize_newtype_struct<T>(&mut self,
                                   _name: &'static str,
                                   value: T) -> Result<(), Self::Error>
        where T: ser::Serialize,
    {
        // Don't serialize the newtype struct in a tuple.
        value.serialize(self)
    }
    fn serialize_newtype_variant<T>(&mut self,
                                    _name: &'static str,
                                    _variant_index: usize,
                                    _variant: &'static str,
                                    value: T) -> Result<(), Self::Error>
        where T: ser::Serialize,
    {
        // Don't serialize the newtype struct variant in a tuple.
        value.serialize(self)
    }
}

impl ser::Serialize for Value {
    fn serialize<E>(&self, e: &mut E) -> Result<(), E::Error>
        where E: ser::Serializer
    {
        match *self {
            Value::String(ref s) => e.serialize_str(s),
            Value::Integer(i) => e.serialize_i64(i),
            Value::Float(f) => e.serialize_f64(f),
            Value::Boolean(b) => e.serialize_bool(b),
            Value::Datetime(ref s) => e.serialize_str(s),
            Value::Array(ref a) => {
                e.serialize_seq(ser::impls::SeqIteratorVisitor::new(a.iter(),
                                                                Some(a.len())))
            }
            Value::Table(ref t) => {
                e.serialize_map(ser::impls::MapIteratorVisitor::new(t.iter(),
                                                                Some(t.len())))
            }
        }
    }
}

impl ser::Error for Error {
    fn custom<T: Into<String>>(msg: T) -> Error {
        Error::Custom(msg.into())
    }
}
