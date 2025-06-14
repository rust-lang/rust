//@ run-pass
#![allow(unused_variables)]
//@ compile-flags: -C codegen-units=8 -O
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

pub fn multispace<T: Copy>(input: T) -> crate::IResult<i8, i8> {
    crate::IResult::Done(0, 0)
}

mod nom_sql {
    fn where_clause(i: &[u8]) -> crate::IResult<&[u8], Option<String>> {
        let X = match crate::multispace(i) {
            crate::IResult::Done(..) => crate::IResult::Done(i, None::<String>),
            _ => crate::IResult::Error(crate::Err::NodePosition(0)),
        };
        match X {
            crate::IResult::Done(_, _) => crate::IResult::Done(i, None),
            _ => X
        }
    }

    pub fn selection(i: &[u8]) {
        let Y = match {
            match {
                where_clause(i)
            } {
                crate::IResult::Done(_, o) => crate::IResult::Done(i, Some(o)),
                crate::IResult::Error(_) => crate::IResult::Done(i, None),
                _ => crate::IResult::Incomplete(0, 0),
            }
        } {
            crate::IResult::Done(z, _) => crate::IResult::Done(z, None::<String>),
            _ => return ()
        };
        match Y {
            crate::IResult::Done(x, _) => {
                let bytes = b";   ";
                let len = x.len();
                bytes[len];
            }
            _ => ()
        }
    }
}
