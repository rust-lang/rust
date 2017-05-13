use rustc_serialize;
use std::mem;
use std::collections::BTreeMap;

use super::{Decoder, DecodeError};
use super::DecodeErrorKind::*;
use Value;

impl rustc_serialize::Decoder for Decoder {
    type Error = DecodeError;
    fn read_nil(&mut self) -> Result<(), DecodeError> {
        match self.toml {
            Some(Value::String(ref s)) if s.is_empty() => {}
            Some(Value::String(..)) => return Err(self.err(NilTooLong)),
            ref found => return Err(self.mismatch("string", found)),
        }
        self.toml.take();
        Ok(())
    }
    fn read_usize(&mut self) -> Result<usize, DecodeError> {
        self.read_i64().map(|i| i as usize)
    }
    fn read_u64(&mut self) -> Result<u64, DecodeError> {
        self.read_i64().map(|i| i as u64)
    }
    fn read_u32(&mut self) -> Result<u32, DecodeError> {
        self.read_i64().map(|i| i as u32)
    }
    fn read_u16(&mut self) -> Result<u16, DecodeError> {
        self.read_i64().map(|i| i as u16)
    }
    fn read_u8(&mut self) -> Result<u8, DecodeError> {
        self.read_i64().map(|i| i as u8)
    }
    fn read_isize(&mut self) -> Result<isize, DecodeError> {
        self.read_i64().map(|i| i as isize)
    }
    fn read_i64(&mut self) -> Result<i64, DecodeError> {
        match self.toml {
            Some(Value::Integer(i)) => { self.toml.take(); Ok(i) }
            ref found => Err(self.mismatch("integer", found)),
        }
    }
    fn read_i32(&mut self) -> Result<i32, DecodeError> {
        self.read_i64().map(|i| i as i32)
    }
    fn read_i16(&mut self) -> Result<i16, DecodeError> {
        self.read_i64().map(|i| i as i16)
    }
    fn read_i8(&mut self) -> Result<i8, DecodeError> {
        self.read_i64().map(|i| i as i8)
    }
    fn read_bool(&mut self) -> Result<bool, DecodeError> {
        match self.toml {
            Some(Value::Boolean(b)) => { self.toml.take(); Ok(b) }
            ref found => Err(self.mismatch("bool", found)),
        }
    }
    fn read_f64(&mut self) -> Result<f64, DecodeError> {
        match self.toml {
            Some(Value::Float(f)) => { self.toml.take(); Ok(f) },
            ref found => Err(self.mismatch("float", found)),
        }
    }
    fn read_f32(&mut self) -> Result<f32, DecodeError> {
        self.read_f64().map(|f| f as f32)
    }
    fn read_char(&mut self) -> Result<char, DecodeError> {
        let ch = match self.toml {
            Some(Value::String(ref s)) if s.chars().count() == 1 =>
                s.chars().next().unwrap(),
            ref found => return Err(self.mismatch("string", found)),
        };
        self.toml.take();
        Ok(ch)
    }
    fn read_str(&mut self) -> Result<String, DecodeError> {
        match self.toml.take() {
            Some(Value::String(s)) => Ok(s),
            found => {
                let err = Err(self.mismatch("string", &found));
                self.toml = found;
                err
            }
        }
    }

    // Compound types:
    fn read_enum<T, F>(&mut self, _name: &str, f: F)
        -> Result<T, DecodeError>
        where F: FnOnce(&mut Decoder) -> Result<T, DecodeError>
    {
        f(self)
    }

    fn read_enum_variant<T, F>(&mut self, names: &[&str], mut f: F)
        -> Result<T, DecodeError>
        where F: FnMut(&mut Decoder, usize) -> Result<T, DecodeError>
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
        for i in 0..names.len() {
            let mut d = self.sub_decoder(self.toml.clone(), "");
            match f(&mut d, i) {
                Ok(t) => {
                    self.toml = d.toml;
                    return Ok(t)
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
        Err(first_error.unwrap_or_else(|| self.err(NoEnumVariants)))
    }
    fn read_enum_variant_arg<T, F>(&mut self, _a_idx: usize, f: F)
        -> Result<T, DecodeError>
        where F: FnOnce(&mut Decoder) -> Result<T, DecodeError>
    {
        f(self)
    }

    fn read_enum_struct_variant<T, F>(&mut self, _names: &[&str], _f: F)
        -> Result<T, DecodeError>
        where F: FnMut(&mut Decoder, usize) -> Result<T, DecodeError>
    {
        panic!()
    }
    fn read_enum_struct_variant_field<T, F>(&mut self,
                                            _f_name: &str,
                                            _f_idx: usize,
                                            _f: F)
        -> Result<T, DecodeError>
        where F: FnOnce(&mut Decoder) -> Result<T, DecodeError>
    {
        panic!()
    }

    fn read_struct<T, F>(&mut self, _s_name: &str, _len: usize, f: F)
        -> Result<T, DecodeError>
        where F: FnOnce(&mut Decoder) -> Result<T, DecodeError>
    {
        match self.toml {
            Some(Value::Table(..)) => {
                let ret = try!(f(self));
                match self.toml {
                    Some(Value::Table(ref t)) if t.is_empty() => {}
                    _ => return Ok(ret)
                }
                self.toml.take();
                Ok(ret)
            }
            ref found => Err(self.mismatch("table", found)),
        }
    }
    fn read_struct_field<T, F>(&mut self, f_name: &str, _f_idx: usize, f: F)
        -> Result<T, DecodeError>
        where F: FnOnce(&mut Decoder) -> Result<T, DecodeError>
    {
        let field = format!("{}", f_name);
        let toml = match self.toml {
            Some(Value::Table(ref mut table)) => {
                table.remove(&field)
                     .or_else(|| table.remove(&f_name.replace("_", "-")))
            },
            ref found => return Err(self.mismatch("table", found)),
        };
        let mut d = self.sub_decoder(toml, f_name);
        let ret = try!(f(&mut d));
        if let Some(value) = d.toml {
            if let Some(Value::Table(ref mut table)) = self.toml {
                table.insert(field, value);
            }
        }
        Ok(ret)
    }

    fn read_tuple<T, F>(&mut self, tuple_len: usize, f: F)
        -> Result<T, DecodeError>
        where F: FnOnce(&mut Decoder) -> Result<T, DecodeError>
    {
        self.read_seq(move |d, len| {
            assert!(len == tuple_len,
                    "expected tuple of length `{}`, found tuple \
                         of length `{}`", tuple_len, len);
            f(d)
        })
    }
    fn read_tuple_arg<T, F>(&mut self, a_idx: usize, f: F)
        -> Result<T, DecodeError>
        where F: FnOnce(&mut Decoder) -> Result<T, DecodeError>
    {
        self.read_seq_elt(a_idx, f)
    }

    fn read_tuple_struct<T, F>(&mut self, _s_name: &str, _len: usize, _f: F)
        -> Result<T, DecodeError>
        where F: FnOnce(&mut Decoder) -> Result<T, DecodeError>
    {
        panic!()
    }
    fn read_tuple_struct_arg<T, F>(&mut self, _a_idx: usize, _f: F)
        -> Result<T, DecodeError>
        where F: FnOnce(&mut Decoder) -> Result<T, DecodeError>
    {
        panic!()
    }

    // Specialized types:
    fn read_option<T, F>(&mut self, mut f: F)
        -> Result<T, DecodeError>
        where F: FnMut(&mut Decoder, bool) -> Result<T, DecodeError>
    {
        match self.toml {
            Some(..) => f(self, true),
            None => f(self, false),
        }
    }

    fn read_seq<T, F>(&mut self, f: F) -> Result<T, DecodeError>
        where F: FnOnce(&mut Decoder, usize) -> Result<T, DecodeError>
    {
        let len = match self.toml {
            Some(Value::Array(ref arr)) => arr.len(),
            None => 0,
            ref found => return Err(self.mismatch("array", found)),
        };
        let ret = try!(f(self, len));
        match self.toml {
            Some(Value::Array(ref mut arr)) => {
                arr.retain(|slot| slot.as_integer() != Some(0));
                if !arr.is_empty() { return Ok(ret) }
            }
            _ => return Ok(ret)
        }
        self.toml.take();
        Ok(ret)
    }
    fn read_seq_elt<T, F>(&mut self, idx: usize, f: F)
        -> Result<T, DecodeError>
        where F: FnOnce(&mut Decoder) -> Result<T, DecodeError>
    {
        let toml = match self.toml {
            Some(Value::Array(ref mut arr)) => {
                mem::replace(&mut arr[idx], Value::Integer(0))
            }
            ref found => return Err(self.mismatch("array", found)),
        };
        let mut d = self.sub_decoder(Some(toml), "");
        let ret = try!(f(&mut d));
        if let Some(toml) = d.toml {
            if let Some(Value::Array(ref mut arr)) = self.toml {
                arr[idx] = toml;
            }
        }
        Ok(ret)
    }

    fn read_map<T, F>(&mut self, f: F)
        -> Result<T, DecodeError>
        where F: FnOnce(&mut Decoder, usize) -> Result<T, DecodeError>
    {
        let map = match self.toml.take() {
            Some(Value::Table(table)) => table,
            found => {
                self.toml = found;
                return Err(self.mismatch("table", &self.toml))
            }
        };
        let amt = map.len();
        let prev_iter = mem::replace(&mut self.cur_map,
                                     map.into_iter().peekable());
        let prev_map = mem::replace(&mut self.leftover_map, BTreeMap::new());
        let ret = try!(f(self, amt));
        let leftover = mem::replace(&mut self.leftover_map, prev_map);
        self.cur_map = prev_iter;
        if !leftover.is_empty() {
            self.toml = Some(Value::Table(leftover));
        }
        Ok(ret)
    }
    fn read_map_elt_key<T, F>(&mut self, idx: usize, f: F)
        -> Result<T, DecodeError>
        where F: FnOnce(&mut Decoder) -> Result<T, DecodeError>
    {
        let key = match self.cur_map.peek().map(|p| p.0.clone()) {
            Some(k) => k,
            None => return Err(self.err(ExpectedMapKey(idx))),
        };
        let val = Value::String(key.clone());
        f(&mut self.sub_decoder(Some(val), &key))
    }
    fn read_map_elt_val<T, F>(&mut self, idx: usize, f: F)
        -> Result<T, DecodeError>
        where F: FnOnce(&mut Decoder) -> Result<T, DecodeError>
    {
        match self.cur_map.next() {
            Some((key, value)) => {
                let mut d = self.sub_decoder(Some(value), &key);
                let ret = f(&mut d);
                if let Some(toml) = d.toml.take() {
                    self.leftover_map.insert(key, toml);
                }
                ret
            }
            None => Err(self.err(ExpectedMapElement(idx))),
        }
    }

    fn error(&mut self, err: &str) -> DecodeError {
        DecodeError {
            field: self.cur_field.clone(),
            kind: ApplicationError(format!("{}", err))
        }
    }
}

#[cfg(test)]
mod tests {
    use rustc_serialize::Decodable;
    use std::collections::HashMap;

    use {Parser, Decoder, Value};

    #[test]
    fn bad_enum_chooses_longest_error() {
        #[derive(RustcDecodable)]
        #[allow(dead_code)]
        struct Foo {
            wut: HashMap<String, Bar>,
        }

        #[derive(RustcDecodable)]
        enum Bar {
            Simple(String),
            Detailed(Baz),
        }

        #[derive(RustcDecodable, Debug)]
        struct Baz {
            features: Vec<String>,
        }

        let s = r#"
            [wut]
            a = { features = "" }
        "#;
        let v = Parser::new(s).parse().unwrap();
        let mut d = Decoder::new(Value::Table(v));
        let err = match Foo::decode(&mut d) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert_eq!(err.field.as_ref().unwrap(), "wut.a.features");

    }
}
