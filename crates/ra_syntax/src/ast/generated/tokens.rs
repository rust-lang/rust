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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RangeSeparator {
    Dotdot(Dotdot),
    Dotdotdot(Dotdotdot),
    Dotdoteq(Dotdoteq),
}
impl From<Dotdot> for RangeSeparator {
    fn from(node: Dotdot) -> RangeSeparator { RangeSeparator::Dotdot(node) }
}
impl From<Dotdotdot> for RangeSeparator {
    fn from(node: Dotdotdot) -> RangeSeparator { RangeSeparator::Dotdotdot(node) }
}
impl From<Dotdoteq> for RangeSeparator {
    fn from(node: Dotdoteq) -> RangeSeparator { RangeSeparator::Dotdoteq(node) }
}
impl std::fmt::Display for RangeSeparator {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl AstToken for RangeSeparator {
    fn can_cast(kind: SyntaxKind) -> bool {
        match kind {
            DOTDOT | DOTDOTDOT | DOTDOTEQ => true,
            _ => false,
        }
    }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        let res = match syntax.kind() {
            DOTDOT => RangeSeparator::Dotdot(Dotdot { syntax }),
            DOTDOTDOT => RangeSeparator::Dotdotdot(Dotdotdot { syntax }),
            DOTDOTEQ => RangeSeparator::Dotdoteq(Dotdoteq { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxToken {
        match self {
            RangeSeparator::Dotdot(it) => &it.syntax,
            RangeSeparator::Dotdotdot(it) => &it.syntax,
            RangeSeparator::Dotdoteq(it) => &it.syntax,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BinOp {
    Pipepipe(Pipepipe),
    Ampamp(Ampamp),
    Eqeq(Eqeq),
    Neq(Neq),
    Lteq(Lteq),
    Gteq(Gteq),
    LAngle(LAngle),
    RAngle(RAngle),
    Plus(Plus),
    Star(Star),
    Minus(Minus),
    Slash(Slash),
    Percent(Percent),
    Shl(Shl),
    Shr(Shr),
    Caret(Caret),
    Pipe(Pipe),
    Amp(Amp),
    Eq(Eq),
    Pluseq(Pluseq),
    Slasheq(Slasheq),
    Stareq(Stareq),
    Percenteq(Percenteq),
    Shreq(Shreq),
    Shleq(Shleq),
    Minuseq(Minuseq),
    Pipeeq(Pipeeq),
    Ampeq(Ampeq),
    Careteq(Careteq),
}
impl From<Pipepipe> for BinOp {
    fn from(node: Pipepipe) -> BinOp { BinOp::Pipepipe(node) }
}
impl From<Ampamp> for BinOp {
    fn from(node: Ampamp) -> BinOp { BinOp::Ampamp(node) }
}
impl From<Eqeq> for BinOp {
    fn from(node: Eqeq) -> BinOp { BinOp::Eqeq(node) }
}
impl From<Neq> for BinOp {
    fn from(node: Neq) -> BinOp { BinOp::Neq(node) }
}
impl From<Lteq> for BinOp {
    fn from(node: Lteq) -> BinOp { BinOp::Lteq(node) }
}
impl From<Gteq> for BinOp {
    fn from(node: Gteq) -> BinOp { BinOp::Gteq(node) }
}
impl From<LAngle> for BinOp {
    fn from(node: LAngle) -> BinOp { BinOp::LAngle(node) }
}
impl From<RAngle> for BinOp {
    fn from(node: RAngle) -> BinOp { BinOp::RAngle(node) }
}
impl From<Plus> for BinOp {
    fn from(node: Plus) -> BinOp { BinOp::Plus(node) }
}
impl From<Star> for BinOp {
    fn from(node: Star) -> BinOp { BinOp::Star(node) }
}
impl From<Minus> for BinOp {
    fn from(node: Minus) -> BinOp { BinOp::Minus(node) }
}
impl From<Slash> for BinOp {
    fn from(node: Slash) -> BinOp { BinOp::Slash(node) }
}
impl From<Percent> for BinOp {
    fn from(node: Percent) -> BinOp { BinOp::Percent(node) }
}
impl From<Shl> for BinOp {
    fn from(node: Shl) -> BinOp { BinOp::Shl(node) }
}
impl From<Shr> for BinOp {
    fn from(node: Shr) -> BinOp { BinOp::Shr(node) }
}
impl From<Caret> for BinOp {
    fn from(node: Caret) -> BinOp { BinOp::Caret(node) }
}
impl From<Pipe> for BinOp {
    fn from(node: Pipe) -> BinOp { BinOp::Pipe(node) }
}
impl From<Amp> for BinOp {
    fn from(node: Amp) -> BinOp { BinOp::Amp(node) }
}
impl From<Eq> for BinOp {
    fn from(node: Eq) -> BinOp { BinOp::Eq(node) }
}
impl From<Pluseq> for BinOp {
    fn from(node: Pluseq) -> BinOp { BinOp::Pluseq(node) }
}
impl From<Slasheq> for BinOp {
    fn from(node: Slasheq) -> BinOp { BinOp::Slasheq(node) }
}
impl From<Stareq> for BinOp {
    fn from(node: Stareq) -> BinOp { BinOp::Stareq(node) }
}
impl From<Percenteq> for BinOp {
    fn from(node: Percenteq) -> BinOp { BinOp::Percenteq(node) }
}
impl From<Shreq> for BinOp {
    fn from(node: Shreq) -> BinOp { BinOp::Shreq(node) }
}
impl From<Shleq> for BinOp {
    fn from(node: Shleq) -> BinOp { BinOp::Shleq(node) }
}
impl From<Minuseq> for BinOp {
    fn from(node: Minuseq) -> BinOp { BinOp::Minuseq(node) }
}
impl From<Pipeeq> for BinOp {
    fn from(node: Pipeeq) -> BinOp { BinOp::Pipeeq(node) }
}
impl From<Ampeq> for BinOp {
    fn from(node: Ampeq) -> BinOp { BinOp::Ampeq(node) }
}
impl From<Careteq> for BinOp {
    fn from(node: Careteq) -> BinOp { BinOp::Careteq(node) }
}
impl std::fmt::Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl AstToken for BinOp {
    fn can_cast(kind: SyntaxKind) -> bool {
        match kind {
            PIPEPIPE | AMPAMP | EQEQ | NEQ | LTEQ | GTEQ | L_ANGLE | R_ANGLE | PLUS | STAR
            | MINUS | SLASH | PERCENT | SHL | SHR | CARET | PIPE | AMP | EQ | PLUSEQ | SLASHEQ
            | STAREQ | PERCENTEQ | SHREQ | SHLEQ | MINUSEQ | PIPEEQ | AMPEQ | CARETEQ => true,
            _ => false,
        }
    }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        let res = match syntax.kind() {
            PIPEPIPE => BinOp::Pipepipe(Pipepipe { syntax }),
            AMPAMP => BinOp::Ampamp(Ampamp { syntax }),
            EQEQ => BinOp::Eqeq(Eqeq { syntax }),
            NEQ => BinOp::Neq(Neq { syntax }),
            LTEQ => BinOp::Lteq(Lteq { syntax }),
            GTEQ => BinOp::Gteq(Gteq { syntax }),
            L_ANGLE => BinOp::LAngle(LAngle { syntax }),
            R_ANGLE => BinOp::RAngle(RAngle { syntax }),
            PLUS => BinOp::Plus(Plus { syntax }),
            STAR => BinOp::Star(Star { syntax }),
            MINUS => BinOp::Minus(Minus { syntax }),
            SLASH => BinOp::Slash(Slash { syntax }),
            PERCENT => BinOp::Percent(Percent { syntax }),
            SHL => BinOp::Shl(Shl { syntax }),
            SHR => BinOp::Shr(Shr { syntax }),
            CARET => BinOp::Caret(Caret { syntax }),
            PIPE => BinOp::Pipe(Pipe { syntax }),
            AMP => BinOp::Amp(Amp { syntax }),
            EQ => BinOp::Eq(Eq { syntax }),
            PLUSEQ => BinOp::Pluseq(Pluseq { syntax }),
            SLASHEQ => BinOp::Slasheq(Slasheq { syntax }),
            STAREQ => BinOp::Stareq(Stareq { syntax }),
            PERCENTEQ => BinOp::Percenteq(Percenteq { syntax }),
            SHREQ => BinOp::Shreq(Shreq { syntax }),
            SHLEQ => BinOp::Shleq(Shleq { syntax }),
            MINUSEQ => BinOp::Minuseq(Minuseq { syntax }),
            PIPEEQ => BinOp::Pipeeq(Pipeeq { syntax }),
            AMPEQ => BinOp::Ampeq(Ampeq { syntax }),
            CARETEQ => BinOp::Careteq(Careteq { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxToken {
        match self {
            BinOp::Pipepipe(it) => &it.syntax,
            BinOp::Ampamp(it) => &it.syntax,
            BinOp::Eqeq(it) => &it.syntax,
            BinOp::Neq(it) => &it.syntax,
            BinOp::Lteq(it) => &it.syntax,
            BinOp::Gteq(it) => &it.syntax,
            BinOp::LAngle(it) => &it.syntax,
            BinOp::RAngle(it) => &it.syntax,
            BinOp::Plus(it) => &it.syntax,
            BinOp::Star(it) => &it.syntax,
            BinOp::Minus(it) => &it.syntax,
            BinOp::Slash(it) => &it.syntax,
            BinOp::Percent(it) => &it.syntax,
            BinOp::Shl(it) => &it.syntax,
            BinOp::Shr(it) => &it.syntax,
            BinOp::Caret(it) => &it.syntax,
            BinOp::Pipe(it) => &it.syntax,
            BinOp::Amp(it) => &it.syntax,
            BinOp::Eq(it) => &it.syntax,
            BinOp::Pluseq(it) => &it.syntax,
            BinOp::Slasheq(it) => &it.syntax,
            BinOp::Stareq(it) => &it.syntax,
            BinOp::Percenteq(it) => &it.syntax,
            BinOp::Shreq(it) => &it.syntax,
            BinOp::Shleq(it) => &it.syntax,
            BinOp::Minuseq(it) => &it.syntax,
            BinOp::Pipeeq(it) => &it.syntax,
            BinOp::Ampeq(it) => &it.syntax,
            BinOp::Careteq(it) => &it.syntax,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PrefixOp {
    Minus(Minus),
    Excl(Excl),
    Star(Star),
}
impl From<Minus> for PrefixOp {
    fn from(node: Minus) -> PrefixOp { PrefixOp::Minus(node) }
}
impl From<Excl> for PrefixOp {
    fn from(node: Excl) -> PrefixOp { PrefixOp::Excl(node) }
}
impl From<Star> for PrefixOp {
    fn from(node: Star) -> PrefixOp { PrefixOp::Star(node) }
}
impl std::fmt::Display for PrefixOp {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl AstToken for PrefixOp {
    fn can_cast(kind: SyntaxKind) -> bool {
        match kind {
            MINUS | EXCL | STAR => true,
            _ => false,
        }
    }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        let res = match syntax.kind() {
            MINUS => PrefixOp::Minus(Minus { syntax }),
            EXCL => PrefixOp::Excl(Excl { syntax }),
            STAR => PrefixOp::Star(Star { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxToken {
        match self {
            PrefixOp::Minus(it) => &it.syntax,
            PrefixOp::Excl(it) => &it.syntax,
            PrefixOp::Star(it) => &it.syntax,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RangeOp {
    Dotdot(Dotdot),
    Dotdoteq(Dotdoteq),
}
impl From<Dotdot> for RangeOp {
    fn from(node: Dotdot) -> RangeOp { RangeOp::Dotdot(node) }
}
impl From<Dotdoteq> for RangeOp {
    fn from(node: Dotdoteq) -> RangeOp { RangeOp::Dotdoteq(node) }
}
impl std::fmt::Display for RangeOp {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl AstToken for RangeOp {
    fn can_cast(kind: SyntaxKind) -> bool {
        match kind {
            DOTDOT | DOTDOTEQ => true,
            _ => false,
        }
    }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        let res = match syntax.kind() {
            DOTDOT => RangeOp::Dotdot(Dotdot { syntax }),
            DOTDOTEQ => RangeOp::Dotdoteq(Dotdoteq { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxToken {
        match self {
            RangeOp::Dotdot(it) => &it.syntax,
            RangeOp::Dotdoteq(it) => &it.syntax,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LiteralToken {
    IntNumber(IntNumber),
    FloatNumber(FloatNumber),
    String(String),
    RawString(RawString),
    ByteString(ByteString),
    RawByteString(RawByteString),
    Char(Char),
    Byte(Byte),
}
impl From<IntNumber> for LiteralToken {
    fn from(node: IntNumber) -> LiteralToken { LiteralToken::IntNumber(node) }
}
impl From<FloatNumber> for LiteralToken {
    fn from(node: FloatNumber) -> LiteralToken { LiteralToken::FloatNumber(node) }
}
impl From<String> for LiteralToken {
    fn from(node: String) -> LiteralToken { LiteralToken::String(node) }
}
impl From<RawString> for LiteralToken {
    fn from(node: RawString) -> LiteralToken { LiteralToken::RawString(node) }
}
impl From<ByteString> for LiteralToken {
    fn from(node: ByteString) -> LiteralToken { LiteralToken::ByteString(node) }
}
impl From<RawByteString> for LiteralToken {
    fn from(node: RawByteString) -> LiteralToken { LiteralToken::RawByteString(node) }
}
impl From<Char> for LiteralToken {
    fn from(node: Char) -> LiteralToken { LiteralToken::Char(node) }
}
impl From<Byte> for LiteralToken {
    fn from(node: Byte) -> LiteralToken { LiteralToken::Byte(node) }
}
impl std::fmt::Display for LiteralToken {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl AstToken for LiteralToken {
    fn can_cast(kind: SyntaxKind) -> bool {
        match kind {
            INT_NUMBER | FLOAT_NUMBER | STRING | RAW_STRING | BYTE_STRING | RAW_BYTE_STRING
            | CHAR | BYTE => true,
            _ => false,
        }
    }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        let res = match syntax.kind() {
            INT_NUMBER => LiteralToken::IntNumber(IntNumber { syntax }),
            FLOAT_NUMBER => LiteralToken::FloatNumber(FloatNumber { syntax }),
            STRING => LiteralToken::String(String { syntax }),
            RAW_STRING => LiteralToken::RawString(RawString { syntax }),
            BYTE_STRING => LiteralToken::ByteString(ByteString { syntax }),
            RAW_BYTE_STRING => LiteralToken::RawByteString(RawByteString { syntax }),
            CHAR => LiteralToken::Char(Char { syntax }),
            BYTE => LiteralToken::Byte(Byte { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxToken {
        match self {
            LiteralToken::IntNumber(it) => &it.syntax,
            LiteralToken::FloatNumber(it) => &it.syntax,
            LiteralToken::String(it) => &it.syntax,
            LiteralToken::RawString(it) => &it.syntax,
            LiteralToken::ByteString(it) => &it.syntax,
            LiteralToken::RawByteString(it) => &it.syntax,
            LiteralToken::Char(it) => &it.syntax,
            LiteralToken::Byte(it) => &it.syntax,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NameRefToken {
    Ident(Ident),
    IntNumber(IntNumber),
}
impl From<Ident> for NameRefToken {
    fn from(node: Ident) -> NameRefToken { NameRefToken::Ident(node) }
}
impl From<IntNumber> for NameRefToken {
    fn from(node: IntNumber) -> NameRefToken { NameRefToken::IntNumber(node) }
}
impl std::fmt::Display for NameRefToken {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(self.syntax(), f)
    }
}
impl AstToken for NameRefToken {
    fn can_cast(kind: SyntaxKind) -> bool {
        match kind {
            IDENT | INT_NUMBER => true,
            _ => false,
        }
    }
    fn cast(syntax: SyntaxToken) -> Option<Self> {
        let res = match syntax.kind() {
            IDENT => NameRefToken::Ident(Ident { syntax }),
            INT_NUMBER => NameRefToken::IntNumber(IntNumber { syntax }),
            _ => return None,
        };
        Some(res)
    }
    fn syntax(&self) -> &SyntaxToken {
        match self {
            NameRefToken::Ident(it) => &it.syntax,
            NameRefToken::IntNumber(it) => &it.syntax,
        }
    }
}
