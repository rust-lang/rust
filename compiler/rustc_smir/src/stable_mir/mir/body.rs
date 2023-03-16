pub struct Body {
    pub blocks: Vec<BasicBlock>,
}

pub struct BasicBlock {
    pub statements: Vec<Statement>,
    pub terminator: Terminator,
}

pub enum Terminator {
    Goto {
        target: usize,
    },
    SwitchInt {
        discr: Operand,
        targets: Vec<SwitchTarget>,
        otherwise: usize,
    },
    Resume,
    Abort,
    Return,
    Unreachable,
    Drop {
        place: Place,
        target: usize,
        unwind: Option<usize>,
    },
    Call {
        func: Operand,
        args: Vec<Operand>,
        destination: Place,
        target: Option<usize>,
        cleanup: Option<usize>,
    },
    Assert {
        cond: Operand,
        expected: bool,
        msg: String,
        target: usize,
        cleanup: Option<usize>,
    },
}

pub enum Statement {
    Assign(Place, Operand),
    Nop,
}

pub enum Operand {
    Copy(Place),
    Move(Place),
    Constant(String),
}

pub struct Place {
    pub local: usize,
}

pub struct SwitchTarget {
    pub value: u128,
    pub target: usize,
}
