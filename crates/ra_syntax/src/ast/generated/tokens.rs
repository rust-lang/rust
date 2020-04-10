//! Generated file, do not edit by hand, see `xtask/src/codegen`

use crate::{
    ast::AstToken,
    SyntaxKind::{self, *},
    SyntaxToken,
};
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Semi {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Semi {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Semi {
    fn can_cast(kind: SyntaxKind) -> bool { kind == SEMI }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Comma {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Comma {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Comma {
    fn can_cast(kind: SyntaxKind) -> bool { kind == COMMA }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LParen {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for LParen {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for LParen {
    fn can_cast(kind: SyntaxKind) -> bool { kind == L_PAREN }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RParen {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for RParen {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for RParen {
    fn can_cast(kind: SyntaxKind) -> bool { kind == R_PAREN }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LCurly {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for LCurly {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for LCurly {
    fn can_cast(kind: SyntaxKind) -> bool { kind == L_CURLY }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RCurly {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for RCurly {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for RCurly {
    fn can_cast(kind: SyntaxKind) -> bool { kind == R_CURLY }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LBrack {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for LBrack {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for LBrack {
    fn can_cast(kind: SyntaxKind) -> bool { kind == L_BRACK }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RBrack {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for RBrack {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for RBrack {
    fn can_cast(kind: SyntaxKind) -> bool { kind == R_BRACK }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LAngle {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for LAngle {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for LAngle {
    fn can_cast(kind: SyntaxKind) -> bool { kind == L_ANGLE }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RAngle {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for RAngle {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for RAngle {
    fn can_cast(kind: SyntaxKind) -> bool { kind == R_ANGLE }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct At {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for At {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for At {
    fn can_cast(kind: SyntaxKind) -> bool { kind == AT }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pound {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Pound {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Pound {
    fn can_cast(kind: SyntaxKind) -> bool { kind == POUND }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tilde {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Tilde {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Tilde {
    fn can_cast(kind: SyntaxKind) -> bool { kind == TILDE }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Question {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Question {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Question {
    fn can_cast(kind: SyntaxKind) -> bool { kind == QUESTION }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dollar {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Dollar {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Dollar {
    fn can_cast(kind: SyntaxKind) -> bool { kind == DOLLAR }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Amp {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Amp {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Amp {
    fn can_cast(kind: SyntaxKind) -> bool { kind == AMP }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pipe {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Pipe {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Pipe {
    fn can_cast(kind: SyntaxKind) -> bool { kind == PIPE }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Plus {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Plus {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Plus {
    fn can_cast(kind: SyntaxKind) -> bool { kind == PLUS }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Star {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Star {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Star {
    fn can_cast(kind: SyntaxKind) -> bool { kind == STAR }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Slash {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Slash {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Slash {
    fn can_cast(kind: SyntaxKind) -> bool { kind == SLASH }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Caret {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Caret {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Caret {
    fn can_cast(kind: SyntaxKind) -> bool { kind == CARET }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Percent {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Percent {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Percent {
    fn can_cast(kind: SyntaxKind) -> bool { kind == PERCENT }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Underscore {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Underscore {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Underscore {
    fn can_cast(kind: SyntaxKind) -> bool { kind == UNDERSCORE }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dot {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Dot {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Dot {
    fn can_cast(kind: SyntaxKind) -> bool { kind == DOT }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dotdot {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Dotdot {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Dotdot {
    fn can_cast(kind: SyntaxKind) -> bool { kind == DOTDOT }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dotdotdot {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Dotdotdot {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Dotdotdot {
    fn can_cast(kind: SyntaxKind) -> bool { kind == DOTDOTDOT }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dotdoteq {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Dotdoteq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Dotdoteq {
    fn can_cast(kind: SyntaxKind) -> bool { kind == DOTDOTEQ }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Colon {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Colon {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Colon {
    fn can_cast(kind: SyntaxKind) -> bool { kind == COLON }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Coloncolon {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Coloncolon {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Coloncolon {
    fn can_cast(kind: SyntaxKind) -> bool { kind == COLONCOLON }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Eq {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Eq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Eq {
    fn can_cast(kind: SyntaxKind) -> bool { kind == EQ }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Eqeq {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Eqeq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Eqeq {
    fn can_cast(kind: SyntaxKind) -> bool { kind == EQEQ }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FatArrow {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for FatArrow {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for FatArrow {
    fn can_cast(kind: SyntaxKind) -> bool { kind == FAT_ARROW }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Excl {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Excl {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Excl {
    fn can_cast(kind: SyntaxKind) -> bool { kind == EXCL }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Neq {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Neq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Neq {
    fn can_cast(kind: SyntaxKind) -> bool { kind == NEQ }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Minus {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Minus {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Minus {
    fn can_cast(kind: SyntaxKind) -> bool { kind == MINUS }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ThinArrow {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for ThinArrow {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for ThinArrow {
    fn can_cast(kind: SyntaxKind) -> bool { kind == THIN_ARROW }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Lteq {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Lteq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Lteq {
    fn can_cast(kind: SyntaxKind) -> bool { kind == LTEQ }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Gteq {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Gteq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Gteq {
    fn can_cast(kind: SyntaxKind) -> bool { kind == GTEQ }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pluseq {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Pluseq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Pluseq {
    fn can_cast(kind: SyntaxKind) -> bool { kind == PLUSEQ }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Minuseq {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Minuseq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Minuseq {
    fn can_cast(kind: SyntaxKind) -> bool { kind == MINUSEQ }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pipeeq {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Pipeeq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Pipeeq {
    fn can_cast(kind: SyntaxKind) -> bool { kind == PIPEEQ }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ampeq {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Ampeq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Ampeq {
    fn can_cast(kind: SyntaxKind) -> bool { kind == AMPEQ }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Careteq {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Careteq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Careteq {
    fn can_cast(kind: SyntaxKind) -> bool { kind == CARETEQ }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Slasheq {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Slasheq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Slasheq {
    fn can_cast(kind: SyntaxKind) -> bool { kind == SLASHEQ }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Stareq {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Stareq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Stareq {
    fn can_cast(kind: SyntaxKind) -> bool { kind == STAREQ }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Percenteq {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Percenteq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Percenteq {
    fn can_cast(kind: SyntaxKind) -> bool { kind == PERCENTEQ }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ampamp {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Ampamp {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Ampamp {
    fn can_cast(kind: SyntaxKind) -> bool { kind == AMPAMP }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pipepipe {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Pipepipe {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Pipepipe {
    fn can_cast(kind: SyntaxKind) -> bool { kind == PIPEPIPE }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shl {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Shl {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Shl {
    fn can_cast(kind: SyntaxKind) -> bool { kind == SHL }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shr {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Shr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Shr {
    fn can_cast(kind: SyntaxKind) -> bool { kind == SHR }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shleq {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Shleq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Shleq {
    fn can_cast(kind: SyntaxKind) -> bool { kind == SHLEQ }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shreq {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Shreq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Shreq {
    fn can_cast(kind: SyntaxKind) -> bool { kind == SHREQ }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IntNumber {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for IntNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for IntNumber {
    fn can_cast(kind: SyntaxKind) -> bool { kind == INT_NUMBER }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FloatNumber {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for FloatNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for FloatNumber {
    fn can_cast(kind: SyntaxKind) -> bool { kind == FLOAT_NUMBER }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Char {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Char {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Char {
    fn can_cast(kind: SyntaxKind) -> bool { kind == CHAR }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Byte {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Byte {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Byte {
    fn can_cast(kind: SyntaxKind) -> bool { kind == BYTE }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct String {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for String {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for String {
    fn can_cast(kind: SyntaxKind) -> bool { kind == STRING }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RawString {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for RawString {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for RawString {
    fn can_cast(kind: SyntaxKind) -> bool { kind == RAW_STRING }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ByteString {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for ByteString {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for ByteString {
    fn can_cast(kind: SyntaxKind) -> bool { kind == BYTE_STRING }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RawByteString {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for RawByteString {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for RawByteString {
    fn can_cast(kind: SyntaxKind) -> bool { kind == RAW_BYTE_STRING }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Error {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Error {
    fn can_cast(kind: SyntaxKind) -> bool { kind == ERROR }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ident {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Ident {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Ident {
    fn can_cast(kind: SyntaxKind) -> bool { kind == IDENT }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Whitespace {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Whitespace {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Whitespace {
    fn can_cast(kind: SyntaxKind) -> bool { kind == WHITESPACE }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Lifetime {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Lifetime {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Lifetime {
    fn can_cast(kind: SyntaxKind) -> bool { kind == LIFETIME }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Comment {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Comment {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Comment {
    fn can_cast(kind: SyntaxKind) -> bool { kind == COMMENT }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shebang {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for Shebang {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for Shebang {
    fn can_cast(kind: SyntaxKind) -> bool { kind == SHEBANG }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LDollar {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for LDollar {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for LDollar {
    fn can_cast(kind: SyntaxKind) -> bool { kind == L_DOLLAR }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RDollar {
    pub(crate) syntax: SyntaxToken,
}
impl std::fmt::Display for RDollar {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.syntax, f)
    }
}
impl AstToken for RDollar {
    fn can_cast(kind: SyntaxKind) -> bool { kind == R_DOLLAR }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        if Self::can_cast(syntax.kind()) {
            Some(Self { syntax })
        } else {
            None
        }
    }
    fn syntax(&self) -> &SyntaxToken { &self.syntax }
}
