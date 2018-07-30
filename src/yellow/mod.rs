mod builder;
mod green;
mod red;
mod syntax;

pub use self::syntax::{SyntaxNode, SyntaxNodeRef};
pub(crate) use self::{
    builder::GreenBuilder,
    green::{GreenNode, GreenNodeBuilder},
    red::RedNode,
    syntax::{SyntaxError, SyntaxRoot},
};
