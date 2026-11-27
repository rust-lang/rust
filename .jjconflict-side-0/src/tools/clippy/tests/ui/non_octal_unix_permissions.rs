//@ignore-target: windows

#![warn(clippy::non_octal_unix_permissions)]
use std::fs::{DirBuilder, File, OpenOptions, Permissions};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};

fn main() {
    let permissions = 0o760;

    // OpenOptionsExt::mode
    let mut options = OpenOptions::new();
    options.mode(440);
    //~^ non_octal_unix_permissions
    options.mode(0o400);
    options.mode(permissions);

    // PermissionsExt::from_mode
    let _permissions = Permissions::from_mode(647);
    //~^ non_octal_unix_permissions
    let _permissions = Permissions::from_mode(0o000);
    let _permissions = Permissions::from_mode(permissions);

    // PermissionsExt::set_mode
    let f = File::create("foo.txt").unwrap();
    let metadata = f.metadata().unwrap();
    let mut permissions = metadata.permissions();

    permissions.set_mode(644);
    //~^ non_octal_unix_permissions
    permissions.set_mode(0o704);
    // no error
    permissions.set_mode(0b111_000_100);

    // DirBuilderExt::mode
    let mut builder = DirBuilder::new();
    builder.mode(755);
    //~^ non_octal_unix_permissions
    builder.mode(0o406);
    // no error
    permissions.set_mode(0b111000100);
}
