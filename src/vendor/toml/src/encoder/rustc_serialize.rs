use std::mem;

use rustc_serialize;
use Value;
use super::{Encoder, Error, State};
use super::Error::*;

impl rustc_serialize::Encoder for Encoder {
    type Error = Error;

    fn emit_nil(&mut self) -> Result<(), Error> { Ok(()) }
    fn emit_usize(&mut self, v: usize) -> Result<(), Error> {
        self.emit_i64(v as i64)
    }
    fn emit_u8(&mut self, v: u8) -> Result<(), Error> {
        self.emit_i64(v as i64)
    }
    fn emit_u16(&mut self, v: u16) -> Result<(), Error> {
        self.emit_i64(v as i64)
    }
    fn emit_u32(&mut self, v: u32) -> Result<(), Error> {
        self.emit_i64(v as i64)
    }
    fn emit_u64(&mut self, v: u64) -> Result<(), Error> {
        self.emit_i64(v as i64)
    }
    fn emit_isize(&mut self, v: isize) -> Result<(), Error> {
        self.emit_i64(v as i64)
    }
    fn emit_i8(&mut self, v: i8) -> Result<(), Error> {
        self.emit_i64(v as i64)
    }
    fn emit_i16(&mut self, v: i16) -> Result<(), Error> {
        self.emit_i64(v as i64)
    }
    fn emit_i32(&mut self, v: i32) -> Result<(), Error> {
        self.emit_i64(v as i64)
    }
    fn emit_i64(&mut self, v: i64) -> Result<(), Error> {
        self.emit_value(Value::Integer(v))
    }
    fn emit_bool(&mut self, v: bool) -> Result<(), Error> {
        self.emit_value(Value::Boolean(v))
    }
    fn emit_f32(&mut self, v: f32) -> Result<(), Error> { self.emit_f64(v as f64) }
    fn emit_f64(&mut self, v: f64) -> Result<(), Error> {
        self.emit_value(Value::Float(v))
    }
    fn emit_char(&mut self, v: char) -> Result<(), Error> {
        self.emit_str(&*format!("{}", v))
    }
    fn emit_str(&mut self, v: &str) -> Result<(), Error> {
        self.emit_value(Value::String(format!("{}", v)))
    }
    fn emit_enum<F>(&mut self, _name: &str, f: F)
        -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        f(self)
    }
    fn emit_enum_variant<F>(&mut self, _v_name: &str, _v_id: usize,
                            _len: usize, f: F) -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        f(self)
    }
    fn emit_enum_variant_arg<F>(&mut self, _a_idx: usize, f: F)
        -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        f(self)
    }
    fn emit_enum_struct_variant<F>(&mut self, _v_name: &str, _v_id: usize,
                                   _len: usize,
                                   _f: F)
        -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        panic!()
    }
    fn emit_enum_struct_variant_field<F>(&mut self,
                                         _f_name: &str,
                                         _f_idx: usize,
                                         _f: F)
        -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        panic!()
    }
    fn emit_struct<F>(&mut self, _name: &str, _len: usize, f: F)
        -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        self.table(f)
    }
    fn emit_struct_field<F>(&mut self, f_name: &str, _f_idx: usize, f: F)
        -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        let old = mem::replace(&mut self.state,
                               State::NextKey(format!("{}", f_name)));
        try!(f(self));
        if self.state != State::Start {
            return Err(NoValue)
        }
        self.state = old;
        Ok(())
    }
    fn emit_tuple<F>(&mut self, len: usize, f: F)
        -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        self.emit_seq(len, f)
    }
    fn emit_tuple_arg<F>(&mut self, idx: usize, f: F)
        -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        self.emit_seq_elt(idx, f)
    }
    fn emit_tuple_struct<F>(&mut self, _name: &str, _len: usize, _f: F)
        -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        unimplemented!()
    }
    fn emit_tuple_struct_arg<F>(&mut self, _f_idx: usize, _f: F)
        -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        unimplemented!()
    }
    fn emit_option<F>(&mut self, f: F)
        -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        f(self)
    }
    fn emit_option_none(&mut self) -> Result<(), Error> {
        self.emit_none()
    }
    fn emit_option_some<F>(&mut self, f: F) -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        f(self)
    }
    fn emit_seq<F>(&mut self, _len: usize, f: F)
        -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        self.seq(f)
    }
    fn emit_seq_elt<F>(&mut self, _idx: usize, f: F)
        -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        f(self)
    }
    fn emit_map<F>(&mut self, len: usize, f: F)
        -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        self.emit_struct("foo", len, f)
    }
    fn emit_map_elt_key<F>(&mut self, _idx: usize, f: F) -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        self.table_key(f)
    }
    fn emit_map_elt_val<F>(&mut self, _idx: usize, f: F) -> Result<(), Error>
        where F: FnOnce(&mut Encoder) -> Result<(), Error>
    {
        f(self)
    }
}

impl rustc_serialize::Encodable for Value {
    fn encode<E>(&self, e: &mut E) -> Result<(), E::Error>
        where E: rustc_serialize::Encoder
    {
        match *self {
            Value::String(ref s) => e.emit_str(s),
            Value::Integer(i) => e.emit_i64(i),
            Value::Float(f) => e.emit_f64(f),
            Value::Boolean(b) => e.emit_bool(b),
            Value::Datetime(ref s) => e.emit_str(s),
            Value::Array(ref a) => {
                e.emit_seq(a.len(), |e| {
                    for item in a {
                        try!(item.encode(e));
                    }
                    Ok(())
                })
            }
            Value::Table(ref t) => {
                e.emit_map(t.len(), |e| {
                    for (i, (key, value)) in t.iter().enumerate() {
                        try!(e.emit_map_elt_key(i, |e| e.emit_str(key)));
                        try!(e.emit_map_elt_val(i, |e| value.encode(e)));
                    }
                    Ok(())
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashSet};
    use rustc_serialize::{self, Encodable, Decodable};

    use {Encoder, Decoder, DecodeError};
    use Value;
    use Value::{Table, Integer, Array, Float};

    macro_rules! encode( ($t:expr) => ({
        let mut e = Encoder::new();
        $t.encode(&mut e).unwrap();
        e.toml
    }) );

    macro_rules! decode( ($t:expr) => ({
        let mut d = Decoder::new($t);
        Decodable::decode(&mut d).unwrap()
    }) );

    macro_rules! map( ($($k:ident, $v:expr),*) => ({
        let mut _m = BTreeMap::new();
        $(_m.insert(stringify!($k).to_string(), $v);)*
        _m
    }) );

    #[test]
    fn smoke() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo { a: isize }

        let v = Foo { a: 2 };
        assert_eq!(encode!(v), map! { a, Integer(2) });
        assert_eq!(v, decode!(Table(encode!(v))));
    }

    #[test]
    fn smoke_hyphen() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo { a_b: isize }

        let v = Foo { a_b: 2 };
        assert_eq!(encode!(v), map! { a_b, Integer(2) });
        assert_eq!(v, decode!(Table(encode!(v))));

        let mut m = BTreeMap::new();
        m.insert("a-b".to_string(), Integer(2));
        assert_eq!(v, decode!(Table(encode!(v))));
    }

    #[test]
    fn nested() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo { a: isize, b: Bar }
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Bar { a: String }

        let v = Foo { a: 2, b: Bar { a: "test".to_string() } };
        assert_eq!(encode!(v),
                   map! {
                       a, Integer(2),
                       b, Table(map! {
                           a, Value::String("test".to_string())
                       })
                   });
        assert_eq!(v, decode!(Table(encode!(v))));
    }

    #[test]
    fn application_decode_error() {
        #[derive(PartialEq, Debug)]
        struct Range10(usize);
        impl Decodable for Range10 {
             fn decode<D: rustc_serialize::Decoder>(d: &mut D) -> Result<Range10, D::Error> {
                 let x: usize = try!(Decodable::decode(d));
                 if x > 10 {
                     Err(d.error("Value out of range!"))
                 } else {
                     Ok(Range10(x))
                 }
             }
        }
        let mut d_good = Decoder::new(Integer(5));
        let mut d_bad1 = Decoder::new(Value::String("not an isize".to_string()));
        let mut d_bad2 = Decoder::new(Integer(11));

        assert_eq!(Ok(Range10(5)), Decodable::decode(&mut d_good));

        let err1: Result<Range10, _> = Decodable::decode(&mut d_bad1);
        assert!(err1.is_err());
        let err2: Result<Range10, _> = Decodable::decode(&mut d_bad2);
        assert!(err2.is_err());
    }

    #[test]
    fn array() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo { a: Vec<isize> }

        let v = Foo { a: vec![1, 2, 3, 4] };
        assert_eq!(encode!(v),
                   map! {
                       a, Array(vec![
                            Integer(1),
                            Integer(2),
                            Integer(3),
                            Integer(4)
                       ])
                   });
        assert_eq!(v, decode!(Table(encode!(v))));
    }

    #[test]
    fn tuple() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo { a: (isize, isize, isize, isize) }

        let v = Foo { a: (1, 2, 3, 4) };
        assert_eq!(encode!(v),
                   map! {
                       a, Array(vec![
                            Integer(1),
                            Integer(2),
                            Integer(3),
                            Integer(4)
                       ])
                   });
        assert_eq!(v, decode!(Table(encode!(v))));
    }

    #[test]
    fn inner_structs_with_options() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo {
            a: Option<Box<Foo>>,
            b: Bar,
        }
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Bar {
            a: String,
            b: f64,
        }

        let v = Foo {
            a: Some(Box::new(Foo {
                a: None,
                b: Bar { a: "foo".to_string(), b: 4.5 },
            })),
            b: Bar { a: "bar".to_string(), b: 1.0 },
        };
        assert_eq!(encode!(v),
                   map! {
                       a, Table(map! {
                           b, Table(map! {
                               a, Value::String("foo".to_string()),
                               b, Float(4.5)
                           })
                       }),
                       b, Table(map! {
                           a, Value::String("bar".to_string()),
                           b, Float(1.0)
                       })
                   });
        assert_eq!(v, decode!(Table(encode!(v))));
    }

    #[test]
    fn hashmap() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo {
            map: BTreeMap<String, isize>,
            set: HashSet<char>,
        }

        let v = Foo {
            map: {
                let mut m = BTreeMap::new();
                m.insert("foo".to_string(), 10);
                m.insert("bar".to_string(), 4);
                m
            },
            set: {
                let mut s = HashSet::new();
                s.insert('a');
                s
            },
        };
        assert_eq!(encode!(v),
            map! {
                map, Table(map! {
                    foo, Integer(10),
                    bar, Integer(4)
                }),
                set, Array(vec![Value::String("a".to_string())])
            }
        );
        assert_eq!(v, decode!(Table(encode!(v))));
    }

    #[test]
    fn tuple_struct() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo(isize, String, f64);

        let v = Foo(1, "foo".to_string(), 4.5);
        assert_eq!(
            encode!(v),
            map! {
                _field0, Integer(1),
                _field1, Value::String("foo".to_string()),
                _field2, Float(4.5)
            }
        );
        assert_eq!(v, decode!(Table(encode!(v))));
    }

    #[test]
    fn table_array() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo { a: Vec<Bar>, }
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Bar { a: isize }

        let v = Foo { a: vec![Bar { a: 1 }, Bar { a: 2 }] };
        assert_eq!(
            encode!(v),
            map! {
                a, Array(vec![
                    Table(map!{ a, Integer(1) }),
                    Table(map!{ a, Integer(2) }),
                ])
            }
        );
        assert_eq!(v, decode!(Table(encode!(v))));
    }

    #[test]
    fn type_errors() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo { bar: isize }

        let mut d = Decoder::new(Table(map! {
            bar, Float(1.0)
        }));
        let a: Result<Foo, DecodeError> = Decodable::decode(&mut d);
        match a {
            Ok(..) => panic!("should not have decoded"),
            Err(e) => {
                assert_eq!(format!("{}", e),
                           "expected a value of type `integer`, but \
                            found a value of type `float` for the key `bar`");
            }
        }
    }

    #[test]
    fn missing_errors() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo { bar: isize }

        let mut d = Decoder::new(Table(map! {
        }));
        let a: Result<Foo, DecodeError> = Decodable::decode(&mut d);
        match a {
            Ok(..) => panic!("should not have decoded"),
            Err(e) => {
                assert_eq!(format!("{}", e),
                           "expected a value of type `integer` for the key `bar`");
            }
        }
    }

    #[test]
    fn parse_enum() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo { a: E }
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        enum E {
            Bar(isize),
            Baz(f64),
            Last(Foo2),
        }
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo2 {
            test: String,
        }

        let v = Foo { a: E::Bar(10) };
        assert_eq!(
            encode!(v),
            map! { a, Integer(10) }
        );
        assert_eq!(v, decode!(Table(encode!(v))));

        let v = Foo { a: E::Baz(10.2) };
        assert_eq!(
            encode!(v),
            map! { a, Float(10.2) }
        );
        assert_eq!(v, decode!(Table(encode!(v))));

        let v = Foo { a: E::Last(Foo2 { test: "test".to_string() }) };
        assert_eq!(
            encode!(v),
            map! { a, Table(map! { test, Value::String("test".to_string()) }) }
        );
        assert_eq!(v, decode!(Table(encode!(v))));
    }

    #[test]
    fn unused_fields() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo { a: isize }

        let v = Foo { a: 2 };
        let mut d = Decoder::new(Table(map! {
            a, Integer(2),
            b, Integer(5)
        }));
        assert_eq!(v, Decodable::decode(&mut d).unwrap());

        assert_eq!(d.toml, Some(Table(map! {
            b, Integer(5)
        })));
    }

    #[test]
    fn unused_fields2() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo { a: Bar }
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Bar { a: isize }

        let v = Foo { a: Bar { a: 2 } };
        let mut d = Decoder::new(Table(map! {
            a, Table(map! {
                a, Integer(2),
                b, Integer(5)
            })
        }));
        assert_eq!(v, Decodable::decode(&mut d).unwrap());

        assert_eq!(d.toml, Some(Table(map! {
            a, Table(map! {
                b, Integer(5)
            })
        })));
    }

    #[test]
    fn unused_fields3() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo { a: Bar }
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Bar { a: isize }

        let v = Foo { a: Bar { a: 2 } };
        let mut d = Decoder::new(Table(map! {
            a, Table(map! {
                a, Integer(2)
            })
        }));
        assert_eq!(v, Decodable::decode(&mut d).unwrap());

        assert_eq!(d.toml, None);
    }

    #[test]
    fn unused_fields4() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo { a: BTreeMap<String, String> }

        let v = Foo { a: map! { a, "foo".to_string() } };
        let mut d = Decoder::new(Table(map! {
            a, Table(map! {
                a, Value::String("foo".to_string())
            })
        }));
        assert_eq!(v, Decodable::decode(&mut d).unwrap());

        assert_eq!(d.toml, None);
    }

    #[test]
    fn unused_fields5() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo { a: Vec<String> }

        let v = Foo { a: vec!["a".to_string()] };
        let mut d = Decoder::new(Table(map! {
            a, Array(vec![Value::String("a".to_string())])
        }));
        assert_eq!(v, Decodable::decode(&mut d).unwrap());

        assert_eq!(d.toml, None);
    }

    #[test]
    fn unused_fields6() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo { a: Option<Vec<String>> }

        let v = Foo { a: Some(vec![]) };
        let mut d = Decoder::new(Table(map! {
            a, Array(vec![])
        }));
        assert_eq!(v, Decodable::decode(&mut d).unwrap());

        assert_eq!(d.toml, None);
    }

    #[test]
    fn unused_fields7() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo { a: Vec<Bar> }
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Bar { a: isize }

        let v = Foo { a: vec![Bar { a: 1 }] };
        let mut d = Decoder::new(Table(map! {
            a, Array(vec![Table(map! {
                a, Integer(1),
                b, Integer(2)
            })])
        }));
        assert_eq!(v, Decodable::decode(&mut d).unwrap());

        assert_eq!(d.toml, Some(Table(map! {
            a, Array(vec![Table(map! {
                b, Integer(2)
            })])
        })));
    }

    #[test]
    fn unused_fields8() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo { a: BTreeMap<String, Bar> }
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Bar { a: isize }

        let v = Foo { a: map! { a, Bar { a: 2 } } };
        let mut d = Decoder::new(Table(map! {
            a, Table(map! {
                a, Table(map! {
                    a, Integer(2),
                    b, Integer(2)
                })
            })
        }));
        assert_eq!(v, Decodable::decode(&mut d).unwrap());

        assert_eq!(d.toml, Some(Table(map! {
            a, Table(map! {
                a, Table(map! {
                    b, Integer(2)
                })
            })
        })));
    }

    #[test]
    fn empty_arrays() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo { a: Vec<Bar> }
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Bar;

        let v = Foo { a: vec![] };
        let mut d = Decoder::new(Table(map! {}));
        assert_eq!(v, Decodable::decode(&mut d).unwrap());
    }

    #[test]
    fn empty_arrays2() {
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Foo { a: Option<Vec<Bar>> }
        #[derive(RustcEncodable, RustcDecodable, PartialEq, Debug)]
        struct Bar;

        let v = Foo { a: None };
        let mut d = Decoder::new(Table(map! {}));
        assert_eq!(v, Decodable::decode(&mut d).unwrap());

        let v = Foo { a: Some(vec![]) };
        let mut d = Decoder::new(Table(map! {
            a, Array(vec![])
        }));
        assert_eq!(v, Decodable::decode(&mut d).unwrap());
    }

    #[test]
    fn round_trip() {
        let toml = r#"
              [test]
              foo = "bar"

              [[values]]
              foo = "baz"

              [[values]]
              foo = "qux"
        "#;

        let value: Value = toml.parse().unwrap();
        let val2 = ::encode_str(&value).parse().unwrap();
        assert_eq!(value, val2);
    }
}
