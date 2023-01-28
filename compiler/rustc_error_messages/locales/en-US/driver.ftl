driver_rlink_unable_to_read = failed to read rlink file: `{$err}`

driver_rlink_wrong_file_type = The input does not look like a .rlink file

driver_rlink_empty_version_number = The input does not contain version number

driver_rlink_encoding_version_mismatch = .rlink file was produced with encoding version `{$version_array}`, but the current version is `{$rlink_version}`

driver_rlink_rustc_version_mismatch = .rlink file was produced by rustc version `{$rustc_version}`, but the current version is `{$current_version}`

driver_rlink_no_a_file = rlink must be a file

driver_unpretty_dump_fail = pretty-print failed to write `{$path}` due to error `{$err}`

driver_ice = the compiler unexpectedly panicked. this is a bug.
driver_ice_bug_report = we would appreciate a bug report: {$bug_report_url}
driver_ice_version = rustc {$version} running on {$triple}
driver_ice_flags = compiler flags: {$flags}
driver_ice_exclude_cargo_defaults = some of the compiler flags provided by cargo are hidden
