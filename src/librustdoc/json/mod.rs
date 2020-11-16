use crate::clean;
use crate::config::{RenderInfo, RenderOptions};
use crate::error::Error;
use crate::formats::cache::Cache;
use crate::formats::FormatRenderer;

use rustc_span::edition::Edition;

#[derive(Clone)]
crate struct JsonRenderer {}

impl FormatRenderer for JsonRenderer {
    fn init(
        _krate: clean::Crate,
        _options: RenderOptions,
        _render_info: RenderInfo,
        _edition: Edition,
        _cache: &mut Cache,
    ) -> Result<(Self, clean::Crate), Error> {
        unimplemented!()
    }

    fn item(&mut self, _item: clean::Item, _cache: &Cache) -> Result<(), Error> {
        unimplemented!()
    }

    fn mod_item_in(
        &mut self,
        _item: &clean::Item,
        _item_name: &str,
        _cache: &Cache,
    ) -> Result<(), Error> {
        unimplemented!()
    }

    fn mod_item_out(&mut self, _item_name: &str) -> Result<(), Error> {
        unimplemented!()
    }

    fn after_krate(&mut self, _krate: &clean::Crate, _cache: &Cache) -> Result<(), Error> {
        unimplemented!()
    }

    fn after_run(&mut self, _diag: &rustc_errors::Handler) -> Result<(), Error> {
        unimplemented!()
    }
}
