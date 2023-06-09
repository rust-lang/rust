use crate::stable_mir::ty::Ty;

#[derive(Clone, Debug)]
pub struct Body {
    pub blocks: Vec<BasicBlock>,
    pub locals: Vec<Ty>,
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
        unwind: UnwindAction,
    },
    Call {
        func: Operand,
        args: Vec<Operand>,
        destination: Place,
        target: Option<usize>,
        unwind: UnwindAction,
    },
    Assert {
        cond: Operand,
        expected: bool,
        msg: AssertMessage,
        target: usize,
        unwind: UnwindAction,
    },
    GeneratorDrop,
}

#[derive(Clone, Debug)]
pub enum UnwindAction {
    Continue,
    Unreachable,
    Terminate,
    Cleanup(usize),
}

#[derive(Clone, Debug)]
pub enum AssertMessage {
    BoundsCheck { len: Operand, index: Operand },
    Overflow(BinOp, Operand, Operand),
    OverflowNeg(Operand),
    DivisionByZero(Operand),
    RemainderByZero(Operand),
    ResumedAfterReturn(GeneratorKind),
    ResumedAfterPanic(GeneratorKind),
    MisalignedPointerDereference { required: Operand, found: Operand },
}

#[derive(Clone, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitXor,
    BitAnd,
    BitOr,
    Shl,
    Shr,
    Eq,
    Lt,
    Le,
    Ne,
    Ge,
    Gt,
    Offset,
}

#[derive(Clone, Debug)]
pub enum UnOp {
    Not,
    Neg,
}

#[derive(Clone, Debug)]
pub enum GeneratorKind {
    Async(AsyncGeneratorKind),
    Gen,
}

#[derive(Clone, Debug)]
pub enum AsyncGeneratorKind {
    Block,
    Closure,
    Fn,
}

#[derive(Clone, Debug)]
pub enum Statement {
    Assign(Place, Rvalue),
    Nop,
}

// FIXME this is incomplete
#[derive(Clone, Debug)]
pub enum Rvalue {
    Use(Operand),
    CheckedBinaryOp(BinOp, Operand, Operand),
    UnaryOp(UnOp, Operand),
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
    pub projection: String,
}

#[derive(Clone, Debug)]
pub struct SwitchTarget {
    pub value: u128,
    pub target: usize,
}
