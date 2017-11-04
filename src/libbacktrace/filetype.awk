# An awk script to determine the type of a file.
/\177ELF\001/ { if (NR == 1) { print "elf32"; exit } }
/\177ELF\002/ { if (NR == 1) { print "elf64"; exit } }
/\114\001/    { if (NR == 1) { print "pecoff"; exit } }
/\144\206/    { if (NR == 1) { print "pecoff"; exit } }
