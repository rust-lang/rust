use syntax::ast;
use syntax::parse::token::{DelimToken, Token};
use syntax::parse::{stream_to_parser_with_base_dir, Directory, ParseSess};
use syntax::symbol::kw;
use syntax::visit::Visitor;
use syntax_pos::Symbol;

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

        while parser.token != Token::Eof {
            if process_if_cfg {
                if !parser.eat_keyword(kw::If) {
                    return Err("Expected `if`");
                }
                parser
                    .parse_attribute(false)
                    .map_err(|_| "Failed to parse attributes")?;
            }

            if !parser.eat(&Token::OpenDelim(DelimToken::Brace)) {
                return Err("Expected an opening brace");
            }

            while parser.token != Token::CloseDelim(DelimToken::Brace) && parser.token != Token::Eof
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

            if !parser.eat(&Token::CloseDelim(DelimToken::Brace)) {
                return Err("Expected a closing brace");
            }

            if parser.eat(&Token::Eof) {
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
