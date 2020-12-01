use rls_data::config::Config;
use rls_data::{
    self, Analysis, CompilationOptions, CratePreludeData, Def, DefKind, Impl, Import, MacroRef,
    Ref, RefKind, Relation,
};
use rls_span::{Column, Row};

#[derive(Debug)]
pub struct Access {
    pub reachable: bool,
    pub public: bool,
}

pub struct Dumper {
    result: Analysis,
    config: Config,
}

impl Dumper {
    pub fn new(config: Config) -> Dumper {
        Dumper { config: config.clone(), result: Analysis::new(config) }
    }

    pub fn analysis(&self) -> &Analysis {
        &self.result
    }
}

impl Dumper {
    pub fn crate_prelude(&mut self, data: CratePreludeData) {
        self.result.prelude = Some(data)
    }

    pub fn compilation_opts(&mut self, data: CompilationOptions) {
        self.result.compilation = Some(data);
    }

    pub fn _macro_use(&mut self, data: MacroRef) {
        if self.config.pub_only || self.config.reachable_only {
            return;
        }
        self.result.macro_refs.push(data);
    }

    pub fn import(&mut self, access: &Access, import: Import) {
        if !access.public && self.config.pub_only || !access.reachable && self.config.reachable_only
        {
            return;
        }
        self.result.imports.push(import);
    }

    pub fn dump_ref(&mut self, data: Ref) {
        if self.config.pub_only || self.config.reachable_only {
            return;
        }
        self.result.refs.push(data);
    }

    pub fn dump_def(&mut self, access: &Access, mut data: Def) {
        if !access.public && self.config.pub_only || !access.reachable && self.config.reachable_only
        {
            return;
        }
        if data.kind == DefKind::Mod && data.span.file_name.to_str().unwrap() != data.value {
            // If the module is an out-of-line definition, then we'll make the
            // definition the first character in the module's file and turn
            // the declaration into a reference to it.
            let rf = Ref { kind: RefKind::Mod, span: data.span, ref_id: data.id };
            self.result.refs.push(rf);
            data.span = rls_data::SpanData {
                file_name: data.value.clone().into(),
                byte_start: 0,
                byte_end: 0,
                line_start: Row::new_one_indexed(1),
                line_end: Row::new_one_indexed(1),
                column_start: Column::new_one_indexed(1),
                column_end: Column::new_one_indexed(1),
            }
        }
        self.result.defs.push(data);
    }

    pub fn dump_relation(&mut self, data: Relation) {
        self.result.relations.push(data);
    }

    pub fn dump_impl(&mut self, data: Impl) {
        self.result.impls.push(data);
    }
}
