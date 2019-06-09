use syntax::ast;
use syntax::parse::token::{DelimToken, TokenKind};
use syntax::parse::{stream_to_parser_with_base_dir, Directory, ParseSess};
use syntax::symbol::kw;
use syntax::visit::Visitor;
use syntax_pos::Symbol;

use crate::attr::MetaVisitor;

pub(crate) struct ModItem {
    pub(crate) item: ast::Item,
}

/// Traverse `cfg_if!` macro and fetch modules.
pub(crate) struct CfgIfVisitor<'a> {
    parse_sess: &'a ParseSess,
    mods: Vec<ModItem>,
    base_dir: Directory<'a>,
}

impl<'a> CfgIfVisitor<'a> {
    pub(crate) fn new(parse_sess: &'a ParseSess, base_dir: Directory<'a>) -> CfgIfVisitor<'a> {
        CfgIfVisitor {
            mods: vec![],
            parse_sess,
            base_dir,
        }
    }

    pub(crate) fn mods(self) -> Vec<ModItem> {
        self.mods
    }
}

impl<'a, 'ast: 'a> Visitor<'ast> for CfgIfVisitor<'a> {
    fn visit_mac(&mut self, mac: &'ast ast::Mac) {
        match self.visit_mac_inner(mac) {
            Ok(()) => (),
            Err(e) => debug!("{}", e),
        }
    }
}

impl<'a, 'ast: 'a> CfgIfVisitor<'a> {
    fn visit_mac_inner(&mut self, mac: &'ast ast::Mac) -> Result<(), &'static str> {
        if mac.node.path != Symbol::intern("cfg_if") {
            return Err("Expected cfg_if");
        }

        let mut parser = stream_to_parser_with_base_dir(
            self.parse_sess,
            mac.node.tts.clone(),
            self.base_dir.clone(),
        );
        parser.cfg_mods = false;
        let mut process_if_cfg = true;

        while parser.token.kind != TokenKind::Eof {
            if process_if_cfg {
                if !parser.eat_keyword(kw::If) {
                    return Err("Expected `if`");
                }
                parser
                    .parse_attribute(false)
                    .map_err(|_| "Failed to parse attributes")?;
            }

            if !parser.eat(&TokenKind::OpenDelim(DelimToken::Brace)) {
                return Err("Expected an opening brace");
            }

            while parser.token != TokenKind::CloseDelim(DelimToken::Brace)
                && parser.token.kind != TokenKind::Eof
            {
                let item = match parser.parse_item() {
                    Ok(Some(item_ptr)) => item_ptr.into_inner(),
                    Ok(None) => continue,
                    Err(mut err) => {
                        err.cancel();
                        parser.sess.span_diagnostic.reset_err_count();
                        return Err(
                            "Expected item inside cfg_if block, but failed to parse it as an item",
                        );
                    }
                };
                if let ast::ItemKind::Mod(..) = item.node {
                    self.mods.push(ModItem { item });
                }
            }

            if !parser.eat(&TokenKind::CloseDelim(DelimToken::Brace)) {
                return Err("Expected a closing brace");
            }

            if parser.eat(&TokenKind::Eof) {
                break;
            }

            if !parser.eat_keyword(kw::Else) {
                return Err("Expected `else`");
            }

            process_if_cfg = parser.token.is_keyword(kw::If);
        }

        Ok(())
    }
}

/// Extracts `path = "foo.rs"` from attributes.
#[derive(Default)]
pub(crate) struct PathVisitor {
    /// A list of path defined in attributes.
    paths: Vec<String>,
}

impl PathVisitor {
    pub(crate) fn paths(self) -> Vec<String> {
        self.paths
    }
}

impl<'ast> MetaVisitor<'ast> for PathVisitor {
    fn visit_meta_name_value(&mut self, meta_item: &'ast ast::MetaItem, lit: &'ast ast::Lit) {
        if meta_item.check_name(Symbol::intern("path")) && lit.node.is_str() {
            self.paths.push(lit_to_str(lit));
        }
    }
}

#[cfg(not(windows))]
fn lit_to_str(lit: &ast::Lit) -> String {
    match lit.node {
        ast::LitKind::Str(symbol, ..) => symbol.to_string(),
        _ => unreachable!(),
    }
}

#[cfg(windows)]
fn lit_to_str(lit: &ast::Lit) -> String {
    match lit.node {
        ast::LitKind::Str(symbol, ..) => symbol.as_str().replace("/", "\\"),
        _ => unreachable!(),
    }
}
