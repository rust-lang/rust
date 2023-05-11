//! A pretty-printer for MIR.

use std::fmt::{Display, Write};

use hir_def::{body::Body, expr::BindingId};
use hir_expand::name::Name;
use la_arena::ArenaMap;

use crate::{
    db::HirDatabase,
    display::HirDisplay,
    mir::{PlaceElem, ProjectionElem, StatementKind, Terminator},
};

use super::{
    AggregateKind, BasicBlockId, BorrowKind, LocalId, MirBody, Operand, Place, Rvalue, UnOp,
};

impl MirBody {
    pub fn pretty_print(&self, db: &dyn HirDatabase) -> String {
        let hir_body = db.body(self.owner);
        let mut ctx = MirPrettyCtx::new(self, &hir_body, db);
        ctx.for_body();
        ctx.result
    }
}

struct MirPrettyCtx<'a> {
    body: &'a MirBody,
    hir_body: &'a Body,
    db: &'a dyn HirDatabase,
    result: String,
    ident: String,
    local_to_binding: ArenaMap<LocalId, BindingId>,
}

macro_rules! w {
    ($dst:expr, $($arg:tt)*) => {
        { let _ = write!($dst, $($arg)*); }
    };
}

macro_rules! wln {
    ($dst:expr) => {
        { let _ = writeln!($dst); }
    };
    ($dst:expr, $($arg:tt)*) => {
        { let _ = writeln!($dst, $($arg)*); }
    };
}

impl Write for MirPrettyCtx<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let mut it = s.split('\n'); // note: `.lines()` is wrong here
        self.write(it.next().unwrap_or_default());
        for line in it {
            self.write_line();
            self.write(line);
        }
        Ok(())
    }
}

enum LocalName {
    Unknown(LocalId),
    Binding(Name, LocalId),
}

impl Display for LocalName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LocalName::Unknown(l) => write!(f, "_{}", u32::from(l.into_raw())),
            LocalName::Binding(n, l) => write!(f, "{n}_{}", u32::from(l.into_raw())),
        }
    }
}

impl<'a> MirPrettyCtx<'a> {
    fn for_body(&mut self) {
        self.with_block(|this| {
            this.locals();
            wln!(this);
            this.blocks();
        });
    }

    fn with_block(&mut self, f: impl FnOnce(&mut MirPrettyCtx<'_>)) {
        self.ident += "    ";
        wln!(self, "{{");
        f(self);
        for _ in 0..4 {
            self.result.pop();
            self.ident.pop();
        }
        wln!(self, "}}");
    }

    fn new(body: &'a MirBody, hir_body: &'a Body, db: &'a dyn HirDatabase) -> Self {
        let local_to_binding = body.binding_locals.iter().map(|(x, y)| (*y, x)).collect();
        MirPrettyCtx {
            body,
            db,
            result: String::new(),
            ident: String::new(),
            local_to_binding,
            hir_body,
        }
    }

    fn write_line(&mut self) {
        self.result.push('\n');
        self.result += &self.ident;
    }

    fn write(&mut self, line: &str) {
        self.result += line;
    }

    fn locals(&mut self) {
        for (id, local) in self.body.locals.iter() {
            wln!(self, "let {}: {};", self.local_name(id), local.ty.display(self.db));
        }
    }

    fn local_name(&self, local: LocalId) -> LocalName {
        match self.local_to_binding.get(local) {
            Some(b) => LocalName::Binding(self.hir_body.bindings[*b].name.clone(), local),
            None => LocalName::Unknown(local),
        }
    }

    fn basic_block_id(&self, basic_block_id: BasicBlockId) -> String {
        format!("'bb{}", u32::from(basic_block_id.into_raw()))
    }

    fn blocks(&mut self) {
        for (id, block) in self.body.basic_blocks.iter() {
            wln!(self);
            w!(self, "{}: ", self.basic_block_id(id));
            self.with_block(|this| {
                for statement in &block.statements {
                    match &statement.kind {
                        StatementKind::Assign(l, r) => {
                            this.place(l);
                            w!(this, " = ");
                            this.rvalue(r);
                            wln!(this, ";");
                        }
                        StatementKind::StorageDead(p) => {
                            wln!(this, "StorageDead({})", this.local_name(*p));
                        }
                        StatementKind::StorageLive(p) => {
                            wln!(this, "StorageLive({})", this.local_name(*p));
                        }
                        StatementKind::Deinit(p) => {
                            w!(this, "Deinit(");
                            this.place(p);
                            wln!(this, ");");
                        }
                        StatementKind::Nop => wln!(this, "Nop;"),
                    }
                }
                match &block.terminator {
                    Some(terminator) => match terminator {
                        Terminator::Goto { target } => {
                            wln!(this, "goto 'bb{};", u32::from(target.into_raw()))
                        }
                        Terminator::SwitchInt { discr, targets } => {
                            w!(this, "switch ");
                            this.operand(discr);
                            w!(this, " ");
                            this.with_block(|this| {
                                for (c, b) in targets.iter() {
                                    wln!(this, "{c} => {},", this.basic_block_id(b));
                                }
                                wln!(this, "_ => {},", this.basic_block_id(targets.otherwise()));
                            });
                        }
                        Terminator::Call { func, args, destination, target, .. } => {
                            w!(this, "Call ");
                            this.with_block(|this| {
                                w!(this, "func: ");
                                this.operand(func);
                                wln!(this, ",");
                                w!(this, "args: [");
                                this.operand_list(args);
                                wln!(this, "],");
                                w!(this, "destination: ");
                                this.place(destination);
                                wln!(this, ",");
                                w!(this, "target: ");
                                match target {
                                    Some(t) => w!(this, "{}", this.basic_block_id(*t)),
                                    None => w!(this, "<unreachable>"),
                                }
                                wln!(this, ",");
                            });
                        }
                        _ => wln!(this, "{:?};", terminator),
                    },
                    None => wln!(this, "<no-terminator>;"),
                }
            })
        }
    }

    fn place(&mut self, p: &Place) {
        fn f(this: &mut MirPrettyCtx<'_>, local: LocalId, projections: &[PlaceElem]) {
            let Some((last, head)) = projections.split_last() else {
                // no projection
                w!(this, "{}", this.local_name(local));
                return;
            };
            match last {
                ProjectionElem::Deref => {
                    w!(this, "(*");
                    f(this, local, head);
                    w!(this, ")");
                }
                ProjectionElem::Field(field) => {
                    let variant_data = field.parent.variant_data(this.db.upcast());
                    let name = &variant_data.fields()[field.local_id].name;
                    match field.parent {
                        hir_def::VariantId::EnumVariantId(e) => {
                            w!(this, "(");
                            f(this, local, head);
                            let variant_name =
                                &this.db.enum_data(e.parent).variants[e.local_id].name;
                            w!(this, " as {}).{}", variant_name, name);
                        }
                        hir_def::VariantId::StructId(_) | hir_def::VariantId::UnionId(_) => {
                            f(this, local, head);
                            w!(this, ".{name}");
                        }
                    }
                }
                ProjectionElem::TupleField(x) => {
                    f(this, local, head);
                    w!(this, ".{}", x);
                }
                ProjectionElem::Index(l) => {
                    f(this, local, head);
                    w!(this, "[{}]", this.local_name(*l));
                }
                x => {
                    f(this, local, head);
                    w!(this, ".{:?}", x);
                }
            }
        }
        f(self, p.local, &p.projection);
    }

    fn operand(&mut self, r: &Operand) {
        match r {
            Operand::Copy(p) | Operand::Move(p) => {
                // MIR at the time of writing doesn't have difference between move and copy, so we show them
                // equally. Feel free to change it.
                self.place(p);
            }
            Operand::Constant(c) => w!(self, "Const({})", c.display(self.db)),
        }
    }

    fn rvalue(&mut self, r: &Rvalue) {
        match r {
            Rvalue::Use(op) => self.operand(op),
            Rvalue::Ref(r, p) => {
                match r {
                    BorrowKind::Shared => w!(self, "&"),
                    BorrowKind::Shallow => w!(self, "&shallow "),
                    BorrowKind::Unique => w!(self, "&uniq "),
                    BorrowKind::Mut { .. } => w!(self, "&mut "),
                }
                self.place(p);
            }
            Rvalue::Aggregate(AggregateKind::Tuple(_), x) => {
                w!(self, "(");
                self.operand_list(x);
                w!(self, ")");
            }
            Rvalue::Aggregate(AggregateKind::Array(_), x) => {
                w!(self, "[");
                self.operand_list(x);
                w!(self, "]");
            }
            Rvalue::Aggregate(AggregateKind::Adt(_, _), x) => {
                w!(self, "Adt(");
                self.operand_list(x);
                w!(self, ")");
            }
            Rvalue::Aggregate(AggregateKind::Union(_, _), x) => {
                w!(self, "Union(");
                self.operand_list(x);
                w!(self, ")");
            }
            Rvalue::Len(p) => {
                w!(self, "Len(");
                self.place(p);
                w!(self, ")");
            }
            Rvalue::Cast(ck, op, ty) => {
                w!(self, "Discriminant({ck:?}");
                self.operand(op);
                w!(self, "{})", ty.display(self.db));
            }
            Rvalue::CheckedBinaryOp(b, o1, o2) => {
                self.operand(o1);
                w!(self, " {b} ");
                self.operand(o2);
            }
            Rvalue::UnaryOp(u, o) => {
                let u = match u {
                    UnOp::Not => "!",
                    UnOp::Neg => "-",
                };
                w!(self, "{u} ");
                self.operand(o);
            }
            Rvalue::Discriminant(p) => {
                w!(self, "Discriminant(");
                self.place(p);
                w!(self, ")");
            }
            Rvalue::ShallowInitBox(op, _) => {
                w!(self, "ShallowInitBox(");
                self.operand(op);
                w!(self, ")");
            }
            Rvalue::CopyForDeref(p) => {
                w!(self, "CopyForDeref(");
                self.place(p);
                w!(self, ")");
            }
        }
    }

    fn operand_list(&mut self, x: &[Operand]) {
        let mut it = x.iter();
        if let Some(first) = it.next() {
            self.operand(first);
            for op in it {
                w!(self, ", ");
                self.operand(op);
            }
        }
    }
}
