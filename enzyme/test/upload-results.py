import json
import datetime
import subprocess
import requests
import argparse
import platform


def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode('ascii').strip()
    except:
        return "N/A"


def get_git_revision_date():
    try:
        time = subprocess.check_output(
            ['git', 'show', '--quiet', '--format=%ct', 'HEAD'], stderr=subprocess.STDOUT).decode('ascii').strip()
        dt = datetime.datetime.fromtimestamp(
            int(time), tz=datetime.timezone.utc)
        return dt
    except:
        return datetime.datetime.now(datetime.timezone.utc)


def extract_results(json):
    result = []
    githash = get_git_revision_hash()
    time = get_git_revision_date().isoformat()
    arch = platform.platform()
    llvm = ".".join(map(lambda x: str(x), json["__version__"]))

    for test in json["tests"]:
        if test["code"] == "PASS":
            compiletime = test["elapsed"]
            name = test["name"]
            pathstr = name.split(" :: ")[1]
            comps = pathstr.split("/")
            mode = comps[1]
            testname = ".".join(comps[2:])

            res = {
                "mode": mode,
                "llvm-version": llvm,
                "test-suite": "Enzyme Unit Tests",
                "commit": githash,
                "test-name": testname,
                "compile-time": compiletime,
                "timestamp": time,
                "platform": arch
            }
            result.append(res)

    return result


parser = argparse.ArgumentParser()
parser.add_argument('file', type=argparse.FileType(
    'r'), help="JSON file containing the benchmark results")
parser.add_argument('-t', '--token', type=str, required=False,
                    help="Token to authorize at graphing backend")
parser.add_argument('-u', '--url', type=str, required=True,
                    help="URL of the graphing backend")

args = parser.parse_args()

json = json.load(args.file)
results = extract_results(json)

if args.token:
    response = requests.post(args.url, json=results,
                             headers={"X-TOKEN": args.token})
    response.raise_for_status()
else:
    response = requests.post(args.url, json=results)
    response.raise_for_status()
