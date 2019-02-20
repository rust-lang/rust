(function(){
var VERSIONS = [];
if (window.location.protocol === "file:" || window.location.host !== "doc.rust-lang.org") {
    return;
}
var version = window.location.pathname.split("/").filter(function(x) {
    return x.length > 0;
})[0];
if (version === "std") {
    version = "stable";
}
var s = document.createElement("select");
for (var i = 0; i < VERSIONS.length; ++i) {
    var entry = document.createElement("option");
    entry.innerText = VERSIONS[i];
    entry.value = VERSIONS[i];
    if (VERSIONS[i] === version) {
        entry.selected = true;
    }
    s.append(entry);
}
s.id = "jump-version";
s.onchange = function() {
    var parts = window.location.pathname.split("/").filter(function(x) {
        return x.length > 0;
    });
    if (parts[0] !== "std") {
        parts.shift();
    }
    window.location.pathname = this.value + "/" + parts.join("/");
};
document.body.appendChild(s);
}());
