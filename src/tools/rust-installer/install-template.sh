#!/bin/bash

# No undefined variables
set -u

init_logging() {
    local _abs_libdir="$1"
    local _logfile="$_abs_libdir/$TEMPLATE_REL_MANIFEST_DIR/install.log"
    rm -f "$_logfile"
    need_ok "failed to remove old installation log"
    touch "$_logfile"
    need_ok "failed to create installation log"
    LOGFILE="$_logfile"
}

log_line() {
    local _line="$1"

    if [ -n "${LOGFILE-}" -a -e "${LOGFILE-}" ]; then
    echo "$_line" >> "$LOGFILE"
    # Ignore errors, which may happen e.g. after the manifest dir is deleted
    fi
}

msg() {
    local _line="install: ${1-}"
    echo "$_line"
    log_line "$_line"
}

verbose_msg() {
    if [ -n "${CFG_VERBOSE-}" ]; then
    msg "${1-}"
    else
    log_line "install: ${1-}"
    fi
}

step_msg() {
    msg
    msg "$1"
    msg
}

verbose_step_msg() {
    if [ -n "${CFG_VERBOSE-}" ]; then
    msg
    msg "$1"
    msg
    else
    log_line ""
    log_line "install: $1"
    log_line ""
    fi
}

warn() {
    local _line="install: WARNING: $1"
    echo "$_line" >&2
    log_line "$_line"
}

err() {
    local _line="install: error: $1"
    echo "$_line" >&2
    log_line "$_line"
    exit 1
}

# A non-user error that is likely to result in a corrupted install
critical_err() {
    local _line="install: error: $1. see logs at '${LOGFILE-}'"
    echo "$_line" >&2
    log_line "$_line"
    exit 1
}

need_ok() {
    if [ $? -ne 0 ]
    then
        err "$1"
    fi
}

critical_need_ok() {
    if [ $? -ne 0 ]
    then
        critical_err "$1"
    fi
}

want_ok() {
    if [ $? -ne 0 ]; then
    warn "$1"
    fi
}

assert_nz() {
    if [ -z "$1" ]; then err "assert_nz $2"; fi
}

need_cmd() {
    if command -v $1 >/dev/null 2>&1
    then verbose_msg "found $1"
    else err "need $1"
    fi
}

run() {
    local _line="\$ $*"
    "$@"
    local _retval=$?
    log_line "$_line"
    return $_retval
}

write_to_file() {
    local _msg="$1"
    local _file="$2"
    local _line="$ echo \"$_msg\" > \"$_file\""
    echo "$_msg" > "$_file"
    local _retval=$?
    log_line "$_line"
    return $_retval
}

append_to_file() {
    local _msg="$1"
    local _file="$2"
    local _line="$ echo \"$_msg\" >> \"$_file\""
    echo "$_msg" >> "$_file"
    local _retval=$?
    log_line "$_line"
    return $_retval
}

make_dir_recursive() {
    local _dir="$1"
    local _line="$ umask 022 && mkdir -p \"$_dir\""
    umask 022 && mkdir -p "$_dir"
    local _retval=$?
    log_line "$_line"
    return $_retval
}

putvar() {
    local t
    local tlen
    eval t=\$$1
    eval tlen=\${#$1}
}

valopt() {
    VAL_OPTIONS="$VAL_OPTIONS $1"

    local op=$1
    local default=$2
    shift
    shift
    local doc="$*"
    if [ $HELP -eq 0 ]
    then
        local uop=$(echo $op | tr 'a-z-' 'A-Z_')
        local v="CFG_${uop}"
        eval $v="$default"
        for arg in $CFG_ARGS
        do
            if echo "$arg" | grep -q -- "--$op="
            then
                local val=$(echo "$arg" | cut -f2 -d=)
                eval $v=$val
            fi
        done
        putvar $v
    else
        if [ -z "$default" ]
        then
            default="<none>"
        fi
        op="${op}=[${default}]"
        printf "    --%-30s %s\n" "$op" "$doc"
    fi
}

opt() {
    BOOL_OPTIONS="$BOOL_OPTIONS $1"

    local op=$1
    local default=$2
    shift
    shift
    local doc="$*"
    local flag=""

    if [ $default -eq 0 ]
    then
        flag="enable"
    else
        flag="disable"
        doc="don't $doc"
    fi

    if [ $HELP -eq 0 ]
    then
        for arg in $CFG_ARGS
        do
            if [ "$arg" = "--${flag}-${op}" ]
            then
                op=$(echo $op | tr 'a-z-' 'A-Z_')
                flag=$(echo $flag | tr 'a-z' 'A-Z')
                local v="CFG_${flag}_${op}"
                eval $v=1
                putvar $v
            fi
        done
    else
        if [ ! -z "${META-}" ]
        then
            op="$op=<$META>"
        fi
        printf "    --%-30s %s\n" "$flag-$op" "$doc"
     fi
}

flag() {
    BOOL_OPTIONS="$BOOL_OPTIONS $1"

    local op=$1
    shift
    local doc="$*"

    if [ $HELP -eq 0 ]
    then
        for arg in $CFG_ARGS
        do
            if [ "$arg" = "--${op}" ]
            then
                op=$(echo $op | tr 'a-z-' 'A-Z_')
                local v="CFG_${op}"
                eval $v=1
                putvar $v
            fi
        done
    else
        if [ ! -z "${META-}" ]
        then
            op="$op=<$META>"
        fi
        printf "    --%-30s %s\n" "$op" "$doc"
     fi
}

validate_opt () {
    for arg in $CFG_ARGS
    do
        local is_arg_valid=0
        for option in $BOOL_OPTIONS
        do
            if test --disable-$option = $arg
            then
                is_arg_valid=1
            fi
            if test --enable-$option = $arg
            then
                is_arg_valid=1
            fi
            if test --$option = $arg
            then
                is_arg_valid=1
            fi
        done
        for option in $VAL_OPTIONS
        do
            if echo "$arg" | grep -q -- "--$option="
            then
                is_arg_valid=1
            fi
        done
        if [ "$arg" = "--help" ]
        then
            echo
            echo "No more help available for Configure options,"
            echo "check the Wiki or join our IRC channel"
            break
        else
            if test $is_arg_valid -eq 0
            then
                err "Option '$arg' is not recognized"
            fi
        fi
    done
}

absolutify() {
    local file_path="$1"
    local file_path_dirname="$(dirname "$file_path")"
    local file_path_basename="$(basename "$file_path")"
    local file_abs_path="$(abs_path "$file_path_dirname")"
    local file_path="$file_abs_path/$file_path_basename"
    # This is the return value
    RETVAL="$file_path"
}

# Prints the absolute path of a directory to stdout
abs_path() {
    local path="$1"
    # Unset CDPATH because it causes havok: it makes the destination unpredictable
    # and triggers 'cd' to print the path to stdout. Route `cd`'s output to /dev/null
    # for good measure.
    (unset CDPATH && cd "$path" > /dev/null && pwd)
}

uninstall_legacy() {
    local _abs_libdir="$1"

    local _uninstalled_something=false

    # Replace commas in legacy manifest list with spaces
    _legacy_manifest_dirs=`echo "$TEMPLATE_LEGACY_MANIFEST_DIRS" | sed "s/,/ /g"`

    # Uninstall from legacy manifests
    local _md
    for _md in $_legacy_manifest_dirs; do
    # First, uninstall from the installation prefix.
    # Errors are warnings - try to rm everything in the manifest even if some fail.
    if [ -f "$_abs_libdir/$_md/manifest" ]
    then

        # iterate through installed manifest and remove files
        local _p;
        while read _p; do
        # the installed manifest contains absolute paths
        msg "removing legacy file $_p"
        if [ -f "$_p" ]
        then
            run rm -f "$_p"
            want_ok "failed to remove $_p"
        else
            warn "supposedly installed file $_p does not exist!"
        fi
        done < "$_abs_libdir/$_md/manifest"

        # If we fail to remove $md below, then the
        # installed manifest will still be full; the installed manifest
        # needs to be empty before install.
        msg "removing legacy manifest $_abs_libdir/$_md/manifest"
        run rm -f "$_abs_libdir/$_md/manifest"
        # For the above reason, this is a hard error
        need_ok "failed to remove installed manifest"

        # Remove $template_rel_manifest_dir directory
        msg "removing legacy manifest dir $_abs_libdir/$_md"
        run rm -R "$_abs_libdir/$_md"
        want_ok "failed to remove $_md"

        _uninstalled_something=true
    fi
    done

    RETVAL="$_uninstalled_something"
}

uninstall_components() {
    local _abs_libdir="$1"
    local _dest_prefix="$2"
    local _components="$3"

    # We're going to start by uninstalling existing components. This
    local _uninstalled_something=false

    # First, try removing any 'legacy' manifests from before
    # rust-installer
    uninstall_legacy "$_abs_libdir"
    assert_nz "$RETVAL", "RETVAL"
    if [ "$RETVAL" = true ]; then
    _uninstalled_something=true;
    fi

    # Load the version of the installed installer
    local _installed_version=
    if [ -f "$abs_libdir/$TEMPLATE_REL_MANIFEST_DIR/rust-installer-version" ]; then
    _installed_version=`cat "$_abs_libdir/$TEMPLATE_REL_MANIFEST_DIR/rust-installer-version"`

    # Sanity check
    if [ ! -n "$_installed_version" ]; then critical_err "rust installer version is empty"; fi
    fi

    # If there's something installed, then uninstall
    if [ -n "$_installed_version" ]; then
    # Check the version of the installed installer
    case "$_installed_version" in

        # If this is a previous version, then upgrade in place to the
        # current version before uninstalling.
        2 )
        # The only change between version 2 -> 3 is that components are placed
        # in subdirectories of the installer tarball. There are no changes
        # to the installed data format, so nothing to do.
        ;;

        # This is the current version. Nothing need to be done except uninstall.
        "$TEMPLATE_RUST_INSTALLER_VERSION")
        ;;

        # If this is an unknown (future) version then bail.
        * )
        echo "The copy of $TEMPLATE_PRODUCT_NAME at $_dest_prefix was installed using an"
        echo "unknown version ($_installed_version) of rust-installer."
        echo "Uninstall it first with the installer used for the original installation"
        echo "before continuing."
        exit 1
        ;;
    esac

    local _md="$_abs_libdir/$TEMPLATE_REL_MANIFEST_DIR"
    local _installed_components="$(cat "$_md/components")"

    # Uninstall (our components only) before reinstalling
    local _available_component
    for _available_component in $_components; do
        local _installed_component
        for _installed_component in $_installed_components; do
        if [ "$_available_component" = "$_installed_component" ]; then
            msg "uninstalling component '$_available_component'"
            local _component_manifest="$_md/manifest-$_installed_component"

            # Sanity check: there should be a component manifest
            if [ ! -f "$_component_manifest" ]; then
            critical_err "installed component '$_installed_component' has no manifest"
            fi

            # Iterate through installed component manifest and remove files
            local _directive
            while read _directive; do

            local _command=`echo $_directive | cut -f1 -d:`
            local _file=`echo $_directive | cut -f2 -d:`

            # Sanity checks
            if [ ! -n "$_command" ]; then critical_err "malformed installation directive"; fi
            if [ ! -n "$_file" ]; then critical_err "malformed installation directive"; fi

            case "$_command" in
                file)
                verbose_msg "removing file $_file"
                if [ -f "$_file" ]; then
                    run rm -f "$_file"
                    want_ok "failed to remove $_file"
                else
                    warn "supposedly installed file $_file does not exist!"
                fi
                ;;

                dir)
                verbose_msg "removing directory $_file"
                run rm -r "$_file"
                want_ok "unable to remove directory $_file"
                ;;

                *)
                critical_err "unknown installation directive"
                ;;
            esac

            done < "$_component_manifest"

            # Remove the installed component manifest
            verbose_msg "removing component manifest $_component_manifest"
            run rm "$_component_manifest"
            # This is a hard error because the installation is unrecoverable
            local _err_cant_r_manifest="failed to remove installed manifest for component"
            critical_need_ok "$_err_cant_r_manifest '$_installed_component'"

            # Update the installed component list
            local _modified_components="$(sed "/^$_installed_component\$/d" "$_md/components")"
            write_to_file "$_modified_components" "$_md/components"
            critical_need_ok "failed to update installed component list"
        fi
        done
    done

    # If there are no remaining components delete the manifest directory,
    # but only if we're doing an uninstall - if we're doing an install,
    # then leave the manifest directory around to hang onto the logs,
    # and any files not managed by the installer.
    if [ -n "${CFG_UNINSTALL-}" ]; then
        local _remaining_components="$(cat "$_md/components")"
        if [ ! -n "$_remaining_components" ]; then
        verbose_msg "removing manifest directory $_md"
        run rm -r "$_md"
        want_ok "failed to remove $_md"

        maybe_unconfigure_ld
        fi
    fi

    _uninstalled_something=true
    fi

    # There's no installed version. If we were asked to uninstall, then that's a problem.
    if [ -n "${CFG_UNINSTALL-}" -a "$_uninstalled_something" = false ]
    then
    err "unable to find installation manifest at $CFG_LIBDIR/$TEMPLATE_REL_MANIFEST_DIR"
    fi
}

install_components() {
    local _src_dir="$1"
    local _abs_libdir="$2"
    local _dest_prefix="$3"
    local _components="$4"

    local _component
    for _component in $_components; do

    msg "installing component '$_component'"

    # The file name of the manifest we're installing from
    local _input_manifest="$_src_dir/$_component/manifest.in"

    # Sanity check: do we have our input manifests?
    if [ ! -f "$_input_manifest" ]; then
        critical_err "manifest for $_component does not exist at $_input_manifest"
    fi

    # The installed manifest directory
    local _md="$_abs_libdir/$TEMPLATE_REL_MANIFEST_DIR"

    # The file name of the manifest we're going to create during install
    local _installed_manifest="$_md/manifest-$_component"

    # Create the installed manifest, which we will fill in with absolute file paths
    touch "$_installed_manifest"
    critical_need_ok "failed to create installed manifest"

    # Add this component to the installed component list
    append_to_file "$_component" "$_md/components"
    critical_need_ok "failed to update components list for $_component"

    # Now install, iterate through the new manifest and copy files
    local _directive
    while read _directive; do

        local _command=`echo $_directive | cut -f1 -d:`
        local _file=`echo $_directive | cut -f2 -d:`

        # Sanity checks
        if [ ! -n "$_command" ]; then critical_err "malformed installation directive"; fi
        if [ ! -n "$_file" ]; then critical_err "malformed installation directive"; fi

        # Decide the destination of the file
        local _file_install_path="$_dest_prefix/$_file"

        if echo "$_file" | grep "^etc/" > /dev/null
        then
        local _f="$(echo "$_file" | sed 's/^etc\///')"
        _file_install_path="$CFG_SYSCONFDIR/$_f"
        fi

        if echo "$_file" | grep "^bin/" > /dev/null
        then
        local _f="$(echo "$_file" | sed 's/^bin\///')"
        _file_install_path="$CFG_BINDIR/$_f"
        fi

        if echo "$_file" | grep "^lib/" > /dev/null
        then
        local _f="$(echo "$_file" | sed 's/^lib\///')"
        _file_install_path="$CFG_LIBDIR/$_f"
        fi

        if echo "$_file" | grep "^share" > /dev/null
        then
        local _f="$(echo "$_file" | sed 's/^share\///')"
        _file_install_path="$CFG_DATADIR/$_f"
        fi

        if echo "$_file" | grep "^share/man/" > /dev/null
        then
        local _f="$(echo "$_file" | sed 's/^share\/man\///')"
        _file_install_path="$CFG_MANDIR/$_f"
        fi

            # HACK: Try to support overriding --docdir.  Paths with the form
            # "share/doc/$product/" can be redirected to a single --docdir
            # path. If the following detects that --docdir has been specified
            # then it will replace everything preceeding the "$product" path
            # component. The problem here is that the combined rust installer
            # contains two "products": rust and cargo; so the contents of those
            # directories will both be dumped into the same directory; and the
            # contents of those directories are _not_ disjoint. Since this feature
            # is almost entirely to support 'make install' anyway I don't expect
            # this problem to be a big deal in practice.
            if [ "$CFG_DOCDIR" != "<default>" ]
            then
            if echo "$_file" | grep "^share/doc/" > /dev/null
            then
            local _f="$(echo "$_file" | sed 's/^share\/doc\/[^/]*\///')"
            _file_install_path="$CFG_DOCDIR/$_f"
            fi
            fi

        # Make sure there's a directory for it
        make_dir_recursive "$(dirname "$_file_install_path")"
        critical_need_ok "directory creation failed"

        # Make the path absolute so we can uninstall it later without
        # starting from the installation cwd
        absolutify "$_file_install_path"
        _file_install_path="$RETVAL"
        assert_nz "$_file_install_path" "file_install_path"

        case "$_command" in
        file )

            verbose_msg "copying file $_file_install_path"

            maybe_backup_path "$_file_install_path"

            if echo "$_file" | grep "^bin/" > /dev/null || test -x "$_src_dir/$_component/$_file"
            then
            run cp "$_src_dir/$_component/$_file" "$_file_install_path"
            run chmod 755 "$_file_install_path"
            else
            run cp "$_src_dir/$_component/$_file" "$_file_install_path"
            run chmod 644 "$_file_install_path"
            fi
            critical_need_ok "file creation failed"

            # Update the manifest
            append_to_file "file:$_file_install_path" "$_installed_manifest"
            critical_need_ok "failed to update manifest"

            ;;

        dir )

            verbose_msg "copying directory $_file_install_path"

            maybe_backup_path "$_file_install_path"

            run cp -R "$_src_dir/$_component/$_file" "$_file_install_path"
            critical_need_ok "failed to copy directory"

                    # Set permissions. 0755 for dirs, 644 for files
                    run chmod -R u+rwX,go+rX,go-w "$_file_install_path"
                    critical_need_ok "failed to set permissions on directory"

            # Update the manifest
            append_to_file "dir:$_file_install_path" "$_installed_manifest"
            critical_need_ok "failed to update manifest"
            ;;

        *)
            critical_err "unknown installation directive"
            ;;
        esac
    done < "$_input_manifest"

    done
}

maybe_configure_ld() {
    local _abs_libdir="$1"

    local _ostype="$(uname -s)"
    assert_nz "$_ostype"  "ostype"

    if [ "$_ostype" = "Linux" -a ! -n "${CFG_DISABLE_LDCONFIG-}" ]; then

    # Fedora-based systems do not configure the dynamic linker to look
    # /usr/local/lib, which is our default installation directory. To
    # make things just work, try to put that directory in
    # /etc/ld.so.conf.d/rust-installer-v1 so ldconfig picks it up.
    # Issue #30.
    #
    # This will get rm'd when the last component is uninstalled in
    # maybe_unconfigure_ld.
    if [ "$_abs_libdir" = "/usr/local/lib" -a -d "/etc/ld.so.conf.d" ]; then
        echo "$_abs_libdir" > "/etc/ld.so.conf.d/rust-installer-v1-$TEMPLATE_REL_MANIFEST_DIR.conf"
        if [ $? -ne 0 ]; then
        # This shouldn't happen if we've gotten this far
        # installing to /usr/local
        warn "failed to update /etc/ld.so.conf.d. this is unexpected"
        fi
    fi

    verbose_msg "running ldconfig"
    if [ -n "${CFG_VERBOSE-}" ]; then
        ldconfig
    else
        ldconfig 2> /dev/null
    fi
    if [ $? -ne 0 ]
    then
            local _warn_s="failed to run ldconfig. this may happen when \
not installing as root. run with --verbose to see the error"
            warn "$_warn_s"
    fi
    fi
}

maybe_unconfigure_ld() {
    local _ostype="$(uname -s)"
    assert_nz "$_ostype"  "ostype"

    if [ "$_ostype" != "Linux" ]; then
    return 0
    fi

    rm "/etc/ld.so.conf.d/rust-installer-v1-$TEMPLATE_REL_MANIFEST_DIR.conf" 2> /dev/null
    # Above may fail since that file may not have been created on install
}

# Doing our own 'install'-like backup that is consistent across platforms
maybe_backup_path() {
    local _file_install_path="$1"

    if [ -e "$_file_install_path" ]; then
    msg "backing up existing file at $_file_install_path"
    run mv -f "$_file_install_path" "$_file_install_path.old"
    critical_need_ok "failed to back up $_file_install_path"
    fi
}

install_uninstaller() {
    local _src_dir="$1"
    local _src_basename="$2"
    local _abs_libdir="$3"

    local _uninstaller="$_abs_libdir/$TEMPLATE_REL_MANIFEST_DIR/uninstall.sh"
    msg "creating uninstall script at $_uninstaller"
    run cp "$_src_dir/$_src_basename" "$_uninstaller"
    critical_need_ok "unable to install uninstaller"
}

do_preflight_sanity_checks() {
    local _src_dir="$1"
    local _dest_prefix="$2"

    # Sanity check: can we can write to the destination?
    verbose_msg "verifying destination is writable"
    make_dir_recursive "$CFG_LIBDIR"
    need_ok "can't write to destination. consider \`sudo\`."
    touch "$CFG_LIBDIR/rust-install-probe" > /dev/null
    if [ $? -ne 0 ]
    then
    err "can't write to destination. consider \`sudo\`."
    fi
    rm "$CFG_LIBDIR/rust-install-probe"
    need_ok "failed to remove install probe"

    # Sanity check: don't install to the directory containing the installer.
    # That would surely cause chaos.
    verbose_msg "verifying destination is not the same as source"
    local _prefix_dir="$(abs_path "$dest_prefix")"
    if [ "$_src_dir" = "$_dest_prefix" -a "${CFG_UNINSTALL-}" != 1 ]; then
    err "cannot install to same directory as installer"
    fi
}

verbose_msg "looking for install programs"
verbose_msg

need_cmd mkdir
need_cmd printf
need_cmd cut
need_cmd grep
need_cmd uname
need_cmd tr
need_cmd sed
need_cmd chmod
need_cmd env
need_cmd pwd

CFG_ARGS="${@:-}"

HELP=0
if [ "${1-}" = "--help" ]
then
    HELP=1
    shift
    echo
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo
else
    verbose_step_msg "processing arguments"
fi

OPTIONS=""
BOOL_OPTIONS=""
VAL_OPTIONS=""

flag uninstall "only uninstall from the installation prefix"
valopt destdir "" "set installation root"
valopt prefix "/usr/local" "set installation prefix"

# Avoid prepending an extra / to the prefix path if there's no destdir
# NB: CFG vars here are undefined when passing --help
if [ -z "${CFG_DESTDIR-}" ]; then
    CFG_DESTDIR_PREFIX="${CFG_PREFIX-}"
else
    CFG_DESTDIR_PREFIX="$CFG_DESTDIR/$CFG_PREFIX"
fi

# NB This isn't quite the same definition as in `configure`.
# just using 'lib' instead of configure's CFG_LIBDIR_RELATIVE
valopt without "" "comma-separated list of components to not install"
valopt components "" "comma-separated list of components to install"
flag list-components "list available components"
valopt sysconfdir "$CFG_DESTDIR_PREFIX/etc" "install system configuration files"
valopt bindir "$CFG_DESTDIR_PREFIX/bin" "install binaries"
valopt libdir "$CFG_DESTDIR_PREFIX/lib" "install libraries"
valopt datadir "$CFG_DESTDIR_PREFIX/share" "install data"
# NB We repeat datadir default value because we don't set CFG_DATADIR in --help
valopt mandir "${CFG_DATADIR-"$CFG_DESTDIR_PREFIX/share"}/man" "install man pages in PATH"
# NB See the docdir handling in install_components for an explanation of this
# weird <default> string
valopt docdir "\<default\>" "install documentation in PATH"
opt ldconfig 1 "run ldconfig after installation (Linux only)"
opt verify 1 "obsolete"
flag verbose "run with verbose output"

if [ $HELP -eq 1 ]
then
    echo
    exit 0
fi

verbose_step_msg "validating arguments"
validate_opt

# Template configuration.
# These names surrounded by '%%` are replaced by sed when generating install.sh
# FIXME: Might want to consider loading this from a file and not generating install.sh

# Rust or Cargo
TEMPLATE_PRODUCT_NAME=%%TEMPLATE_PRODUCT_NAME%%
# rustlib or cargo
TEMPLATE_REL_MANIFEST_DIR=%%TEMPLATE_REL_MANIFEST_DIR%%
# 'Rust is ready to roll.' or 'Cargo is cool to cruise.'
TEMPLATE_SUCCESS_MESSAGE=%%TEMPLATE_SUCCESS_MESSAGE%%
# Locations to look for directories containing legacy, pre-versioned manifests
TEMPLATE_LEGACY_MANIFEST_DIRS=%%TEMPLATE_LEGACY_MANIFEST_DIRS%%
# The installer version
TEMPLATE_RUST_INSTALLER_VERSION=%%TEMPLATE_RUST_INSTALLER_VERSION%%

# OK, let's get installing ...

# This is where we are installing from
src_dir="$(abs_path $(dirname "$0"))"

# The name of the script
src_basename="$(basename "$0")"

# If we've been run as 'uninstall.sh' (from the existing installation)
# then we're doing a full uninstall, as opposed to the --uninstall flag
# which just means 'uninstall my components'.
if [ "$src_basename" = "uninstall.sh" ]; then
    if [ "${*:-}" != "" ]; then
    # Currently don't know what to do with arguments in this mode
    err "uninstall.sh does not take any arguments"
    fi
    CFG_UNINSTALL=1
    CFG_DESTDIR_PREFIX="$(abs_path "$src_dir/../../")"
    CFG_LIBDIR="$(abs_path "$src_dir/../")"
fi

# This is where we are installing to
dest_prefix="$CFG_DESTDIR_PREFIX"

# Open the components file to get the list of components to install.
# NB: During install this components file is read from the installer's
# source dir, during a full uninstall it's read from the manifest dir,
# and thus contains all installed components.
components=`cat "$src_dir/components"`

# Sanity check: do we have components?
if [ ! -n "$components" ]; then
    err "unable to find installation components"
fi

# If the user asked for a component list, do that and exit
if [ -n "${CFG_LIST_COMPONENTS-}" ]; then
    echo
    echo "# Available components"
    echo
    for component in $components; do
    echo "* $component"
    done
    echo
    exit 0
fi

# If the user specified which components to install/uninstall,
# then validate that they exist and select them for installation
if [ -n "$CFG_COMPONENTS" ]; then
    # Remove commas
    user_components="$(echo "$CFG_COMPONENTS" | sed "s/,/ /g")"
    for user_component in $user_components; do
    found=false
    for my_component in $components; do
        if [ "$user_component" = "$my_component" ]; then
        found=true
        fi
    done
    if [ "$found" = false ]; then
        err "unknown component: $user_component"
    fi
    done
    components="$user_components"
fi

if [ -n "$CFG_WITHOUT" ]; then
    without_components="$(echo "$CFG_WITHOUT" | sed "s/,/ /g")"

    # This does **not** check that all components in without_components are
    # actually present in the list of available components.
    #
    # Currently that's considered good as it makes it easier to be compatible
    # with multiple Rust versions (which may change the exact list of
    # components) when writing install scripts.
    new_comp=""
    for component in $components; do
        found=false
        for my_component in $without_components; do
            if [ "$component" = "$my_component" ]; then
                found=true
            fi
        done
        if [ "$found" = false ]; then
            # If we didn't find the component in without, then add it to new list.
            new_comp="$new_comp $component"
        fi
    done
    components="$new_comp"
fi

if [ -z "$components" ]; then
    if [ -z "${CFG_UNINSTALL-}" ]; then
    err "no components selected for installation"
    else
    err "no components selected for uninstallation"
    fi
fi

do_preflight_sanity_checks "$src_dir" "$dest_prefix"

# Using an absolute path to libdir in a few places so that the status
# messages are consistently using absolute paths.
absolutify "$CFG_LIBDIR"
abs_libdir="$RETVAL"
assert_nz "$abs_libdir" "abs_libdir"

# Create the manifest directory, where we will put our logs
make_dir_recursive "$abs_libdir/$TEMPLATE_REL_MANIFEST_DIR"
need_ok "failed to create $TEMPLATE_REL_MANIFEST_DIR"

# Log messages and commands
init_logging "$abs_libdir"

# First do any uninstallation, including from legacy manifests. This
# will also upgrade the metadata of existing installs.
uninstall_components "$abs_libdir" "$dest_prefix" "$components"

# If we're only uninstalling then exit
if [ -n "${CFG_UNINSTALL-}" ]
then
    echo
    echo "    $TEMPLATE_PRODUCT_NAME is uninstalled."
    echo
    exit 0
fi

# Create the manifest directory again! uninstall_legacy
# may have deleted it.
make_dir_recursive "$abs_libdir/$TEMPLATE_REL_MANIFEST_DIR"
need_ok "failed to create $TEMPLATE_REL_MANIFEST_DIR"

# Drop the version number into the manifest dir
write_to_file "$TEMPLATE_RUST_INSTALLER_VERSION" \
"$abs_libdir/$TEMPLATE_REL_MANIFEST_DIR/rust-installer-version"

critical_need_ok "failed to write installer version"

# Install the uninstaller
install_uninstaller "$src_dir" "$src_basename" "$abs_libdir"

# Install each component
install_components "$src_dir" "$abs_libdir" "$dest_prefix" "$components"

# Make dynamic libraries available to the linker
maybe_configure_ld "$abs_libdir"

echo
echo "    $TEMPLATE_SUCCESS_MESSAGE"
echo
