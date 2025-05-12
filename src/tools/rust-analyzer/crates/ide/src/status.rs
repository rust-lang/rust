use ide_db::RootDatabase;
use ide_db::base_db::{BuiltCrateData, ExtraCrateData};
use itertools::Itertools;
use span::FileId;
use stdx::format_to;

// Feature: Status
//
// Shows internal statistic about memory usage of rust-analyzer.
//
// | Editor  | Action Name |
// |---------|-------------|
// | VS Code | **rust-analyzer: Status** |
//
// ![Status](https://user-images.githubusercontent.com/48062697/113065584-05f34500-91b1-11eb-98cc-5c196f76be7f.gif)
pub(crate) fn status(db: &RootDatabase, file_id: Option<FileId>) -> String {
    let mut buf = String::new();

    // format_to!(buf, "{}\n", collect_query(CompressedFileTextQuery.in_db(db)));
    // format_to!(buf, "{}\n", collect_query(ParseQuery.in_db(db)));
    // format_to!(buf, "{}\n", collect_query(ParseMacroExpansionQuery.in_db(db)));
    // format_to!(buf, "{}\n", collect_query(LibrarySymbolsQuery.in_db(db)));
    // format_to!(buf, "{}\n", collect_query(ModuleSymbolsQuery.in_db(db)));
    // format_to!(buf, "{} in total\n", memory_usage());

    // format_to!(buf, "\nDebug info:\n");
    // format_to!(buf, "{}\n", collect_query(AttrsQuery.in_db(db)));
    // format_to!(buf, "{} ast id maps\n", collect_query_count(AstIdMapQuery.in_db(db)));
    // format_to!(buf, "{} block def maps\n", collect_query_count(BlockDefMapQuery.in_db(db)));

    if let Some(file_id) = file_id {
        format_to!(buf, "\nCrates for file {}:\n", file_id.index());
        let crates = crate::parent_module::crates_for(db, file_id);
        if crates.is_empty() {
            format_to!(buf, "Does not belong to any crate");
        }
        for crate_id in crates {
            let BuiltCrateData {
                root_file_id,
                edition,
                dependencies,
                origin,
                is_proc_macro,
                proc_macro_cwd,
            } = crate_id.data(db);
            let ExtraCrateData { version, display_name, potential_cfg_options } =
                crate_id.extra_data(db);
            let cfg_options = crate_id.cfg_options(db);
            let env = crate_id.env(db);
            format_to!(
                buf,
                "Crate: {}\n",
                match display_name {
                    Some(it) => format!("{it}({crate_id:?})"),
                    None => format!("{crate_id:?}"),
                }
            );
            format_to!(buf, "    Root module file id: {}\n", root_file_id.index());
            format_to!(buf, "    Edition: {}\n", edition);
            format_to!(buf, "    Version: {}\n", version.as_deref().unwrap_or("n/a"));
            format_to!(buf, "    Enabled cfgs: {:?}\n", cfg_options);
            format_to!(buf, "    Potential cfgs: {:?}\n", potential_cfg_options);
            format_to!(buf, "    Env: {:?}\n", env);
            format_to!(buf, "    Origin: {:?}\n", origin);
            format_to!(buf, "    Is a proc macro crate: {}\n", is_proc_macro);
            format_to!(buf, "    Proc macro cwd: {:?}\n", proc_macro_cwd);
            let deps = dependencies
                .iter()
                .map(|dep| format!("{}={:?}", dep.name, dep.crate_id))
                .format(", ");
            format_to!(buf, "    Dependencies: {}\n", deps);
        }
    }

    buf.trim().to_owned()
}
