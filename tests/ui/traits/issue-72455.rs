//@ check-pass

pub trait ResultExt {
    type Ok;
    fn err_eprint_and_ignore(self) -> Option<Self::Ok>;
}

impl<O, E> ResultExt for std::result::Result<O, E>
where
    E: std::error::Error,
{
    type Ok = O;
    fn err_eprint_and_ignore(self) -> Option<O>
    where
        Self: ,
    {
        match self {
            Err(e) => {
                eprintln!("{}", e);
                None
            }
            Ok(o) => Some(o),
        }
    }
}

fn main() {}
