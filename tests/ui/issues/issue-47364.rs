// run-pass
#![allow(unused_variables)]
// compile-flags: -C codegen-units=8 -O
#![allow(non_snake_case)]

fn main() {
    nom_sql::selection(b"x ");
}

pub enum Err<P>{
    Position(P),
    NodePosition(u32),
}

pub enum IResult<I,O> {
    Done(I,O),
    Error(Err<I>),
    Incomplete(u32, u64)
}

pub fn multispace<T: Copy>(input: T) -> ::IResult<i8, i8> {
    ::IResult::Done(0, 0)
}

mod nom_sql {
    fn where_clause(i: &[u8]) -> ::IResult<&[u8], Option<String>> {
        let X = match ::multispace(i) {
            ::IResult::Done(..) => ::IResult::Done(i, None::<String>),
            _ => ::IResult::Error(::Err::NodePosition(0)),
        };
        match X {
            ::IResult::Done(_, _) => ::IResult::Done(i, None),
            _ => X
        }
    }

    pub fn selection(i: &[u8]) {
        let Y = match {
            match {
                where_clause(i)
            } {
                ::IResult::Done(_, o) => ::IResult::Done(i, Some(o)),
                ::IResult::Error(_) => ::IResult::Done(i, None),
                _ => ::IResult::Incomplete(0, 0),
            }
        } {
            ::IResult::Done(z, _) => ::IResult::Done(z, None::<String>),
            _ => return ()
        };
        match Y {
            ::IResult::Done(x, _) => {
                let bytes = b";   ";
                let len = x.len();
                bytes[len];
            }
            _ => ()
        }
    }
}
