pub enum ErrorHandled {
    Reported,
    TooGeneric,
}

impl ErrorHandled {
    pub fn assert_reported(self) {
        match self {
            ErrorHandled::Reported => {}
            ErrorHandled::TooGeneric => panic!(),
        }
    }
}

fn struct_generic(x: Vec<i32>) {
    for v in x {
        println!("{}", v);
    }
    }
} //~ ERROR unexpected closing delimiter: `}`
