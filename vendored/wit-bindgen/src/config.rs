use crate::{LookupItem, lookup_keys};
use anyhow::Result;
use wit_parser::{Function, FunctionKind, Resolve, WorldKey};

bitflags::bitflags! {
    #[derive(Default, Copy, Clone, Debug)]
    pub struct FunctionFlags: u8 {
        const ASYNC = 1 << 0;
        const TRAPPABLE = 1 << 1;
        const STORE = 1 << 2;
        const TRACING = 1 << 3;
        const VERBOSE_TRACING = 1 << 4;
        const IGNORE_WIT = 1 << 5;
        const EXACT = 1 << 6;
    }
}

#[derive(Default, Debug, Clone)]
pub struct FunctionConfig {
    rules: Vec<FunctionRule>,
    pub(crate) default: FunctionFlags,
}

#[derive(Debug, Clone)]
struct FunctionRule {
    filter: String,
    flags: FunctionFlags,
    used: bool,
}

#[derive(Debug, Clone)]
pub enum FunctionFilter {
    Name(String),
    Default,
}

impl FunctionConfig {
    /// Creates a blank set of configuration.
    pub fn new() -> FunctionConfig {
        FunctionConfig::default()
    }

    /// Adds a new rule to this configuration.
    ///
    /// Note that the order rules are added is significant as only the first
    /// matching rule is used for a function.
    pub fn push(&mut self, filter: FunctionFilter, flags: FunctionFlags) {
        match filter {
            FunctionFilter::Name(filter) => {
                self.rules.push(FunctionRule {
                    filter,
                    flags,
                    used: false,
                });
            }
            FunctionFilter::Default => {
                self.default = flags;
            }
        }
    }

    /// Returns the set of configuration flags associated with `func`.
    ///
    /// The `name` provided should include the full name of the function
    /// including its interface. The `kind` is the classification of the
    /// function in WIT which affects the default set of flags.
    pub(crate) fn flags(
        &mut self,
        resolve: &Resolve,
        ns: Option<&WorldKey>,
        func: &Function,
    ) -> FunctionFlags {
        let mut wit_flags = FunctionFlags::empty();

        // If the kind is async, then set the async/store flags as that's a
        // concurrent function which requires access to both.
        match &func.kind {
            FunctionKind::Freestanding
            | FunctionKind::Method(_)
            | FunctionKind::Static(_)
            | FunctionKind::Constructor(_) => {}

            FunctionKind::AsyncFreestanding
            | FunctionKind::AsyncMethod(_)
            | FunctionKind::AsyncStatic(_) => {
                wit_flags |= FunctionFlags::ASYNC | FunctionFlags::STORE;
            }
        }

        let mut ret = FunctionFlags::empty();
        self.add_function_flags(resolve, ns, &func.name, &mut ret);
        if !ret.contains(FunctionFlags::IGNORE_WIT) {
            ret |= wit_flags;
        }
        ret
    }

    pub(crate) fn resource_drop_flags(
        &mut self,
        resolve: &Resolve,
        ns: Option<&WorldKey>,
        resource_name: &str,
    ) -> FunctionFlags {
        let mut ret = FunctionFlags::empty();
        self.add_function_flags(resolve, ns, &format!("[drop]{resource_name}"), &mut ret);
        ret
    }

    fn add_function_flags(
        &mut self,
        resolve: &Resolve,
        key: Option<&WorldKey>,
        name: &str,
        base: &mut FunctionFlags,
    ) {
        let mut apply_rules = |name: &str, is_exact: bool| {
            for rule in self.rules.iter_mut() {
                if name != rule.filter {
                    continue;
                }
                if !is_exact && rule.flags.contains(FunctionFlags::EXACT) {
                    continue;
                }
                rule.used = true;
                *base |= rule.flags;

                // only the first rule is used.
                return true;
            }

            false
        };
        match key {
            Some(key) => {
                for (lookup, projection) in lookup_keys(resolve, key, LookupItem::Name(name)) {
                    if apply_rules(&lookup, projection.is_empty()) {
                        return;
                    }
                }
            }
            None => {
                if apply_rules(name, true) {
                    return;
                }
            }
        }

        *base |= self.default;
    }

    pub(crate) fn assert_all_rules_used(&self, kind: &str) -> Result<()> {
        let mut unused = Vec::new();
        for rule in self.rules.iter().filter(|r| !r.used) {
            unused.push(format!("{:?}: {:?}", rule.filter, rule.flags));
        }

        if unused.is_empty() {
            return Ok(());
        }

        anyhow::bail!("unused `{kind}` rules found: {unused:?}");
    }
}
