#[derive(Clone, Debug)]
pub struct Body {
    pub blocks: Vec<BasicBlock>,
}

#[derive(Clone, Debug)]
pub struct BasicBlock {
    pub statements: Vec<Statement>,
    pub terminator: Terminator,
}

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
pub enum Statement {
    Assign(Place, Operand),
    Nop,
}

#[derive(Clone, Debug)]
pub enum Operand {
    Copy(Place),
    Move(Place),
    Constant(String),
}

#[derive(Clone, Debug)]
pub struct Place {
    pub local: usize,
}

#[derive(Clone, Debug)]
pub struct SwitchTarget {
    pub value: u128,
    pub target: usize,
}
