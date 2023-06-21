use crate::util::*;
use anyhow::{Context, Result};
use std::io::Write;

const TEMPLATE: &str = include_str!("../install-template.sh");

actor! {
    #[derive(Debug)]
    pub struct Scripter {
        /// The name of the product, for display
        #[arg(value_name = "NAME")]
        product_name: String = "Product",

        /// The directory under lib/ where the manifest lives
        #[arg(value_name = "DIR")]
        rel_manifest_dir: String = "manifestlib",

        /// The string to print after successful installation
        #[arg(value_name = "MESSAGE")]
        success_message: String = "Installed.",

        /// Places to look for legacy manifests to uninstall
        #[arg(value_name = "DIRS")]
        legacy_manifest_dirs: String = "",

        /// The name of the output script
        #[arg(value_name = "FILE")]
        output_script: String = "install.sh",
    }
}

impl Scripter {
    /// Generates the actual installer script
    pub fn run(self) -> Result<()> {
        // Replace dashes in the success message with spaces (our arg handling botches spaces)
        // TODO: still needed? Kept for compatibility for now.
        let product_name = self.product_name.replace('-', " ");

        // Replace dashes in the success message with spaces (our arg handling botches spaces)
        // TODO: still needed? Kept for compatibility for now.
        let success_message = self.success_message.replace('-', " ");

        let script = TEMPLATE
            .replace("%%TEMPLATE_PRODUCT_NAME%%", &sh_quote(&product_name))
            .replace("%%TEMPLATE_REL_MANIFEST_DIR%%", &self.rel_manifest_dir)
            .replace("%%TEMPLATE_SUCCESS_MESSAGE%%", &sh_quote(&success_message))
            .replace(
                "%%TEMPLATE_LEGACY_MANIFEST_DIRS%%",
                &sh_quote(&self.legacy_manifest_dirs),
            )
            .replace(
                "%%TEMPLATE_RUST_INSTALLER_VERSION%%",
                &sh_quote(&crate::RUST_INSTALLER_VERSION),
            );

        create_new_executable(&self.output_script)?
            .write_all(script.as_ref())
            .with_context(|| format!("failed to write output script '{}'", self.output_script))?;

        Ok(())
    }
}

fn sh_quote<T: ToString>(s: &T) -> String {
    // We'll single-quote the whole thing, so first replace single-quotes with
    // '"'"' (leave quoting, double-quote one `'`, re-enter single-quoting)
    format!("'{}'", s.to_string().replace('\'', r#"'"'"'"#))
}
