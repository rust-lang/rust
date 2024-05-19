import sys


def check_duplicates():
    auto_content = ""
    manual_content = ""

    with open("src/intrinsic/llvm.rs", "r", encoding="utf8") as f:
        manual_content = f.read()
    with open("src/intrinsic/archs.rs", "r", encoding="utf8") as f:
        auto_content = f.read()

    intrinsics_map = {}
    for line in auto_content.splitlines():
        line = line.strip()
        if not line.startswith('"'):
            continue
        parts = line.split('"')
        if len(parts) != 5:
            continue
        intrinsics_map[parts[1]] = parts[3]

    if len(intrinsics_map) == 0:
        print("No intrinsics found in auto code... Aborting.")
        return 1
    print("Found {} intrinsics in auto code".format(len(intrinsics_map)))
    errors = []
    lines = manual_content.splitlines()
    pos = 0
    found = 0
    while pos < len(lines):
        line = lines[pos].strip()
        # This is our marker.
        if line == "let gcc_name = match name {":
            while pos < len(lines):
                line = lines[pos].strip()
                pos += 1
                if line == "};":
                    # We're done!
                    if found == 0:
                        print("No intrinsics found in manual code even though we found the "
                            "marker... Aborting...")
                        return 1
                    for error in errors:
                        print("ERROR => {}".format(error))
                    return 1 if len(errors) != 0 else 0
                parts = line.split('"')
                if len(parts) != 5:
                    continue
                found += 1
                if parts[1] in intrinsics_map:
                    if parts[3] != intrinsics_map[parts[1]]:
                        print("Same intrinsics (`{}` at line {}) but different GCC "
                            "translations: `{}` != `{}`".format(
                                parts[1], pos, intrinsics_map[parts[1]], parts[3]))
                    else:
                        errors.append("Duplicated intrinsics: `{}` at line {}. Please remove it "
                            " from manual code".format(parts[1], pos))
            # Weird but whatever...
            return 1 if len(errors) != 0 else 0
        pos += 1
    print("No intrinsics found in manual code... Aborting")
    return 1


if __name__ == "__main__":
    sys.exit(check_duplicates())
